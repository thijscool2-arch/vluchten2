"""Inspectie, gecontroleerde koppelingen en temporele voorspelling.
Originele projectcode; bronnen en methoden staan in SOURCES.md en README.md.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent
GROUPS = ['Regionaal', 'Narrowbody', 'Widebody']
WIDE = set('A310 A332 A333 A339 A343 A359 A35K A388 B744 B763 B764 B772 B77L B77W B788 B789 B78X'.split())
NARROW = set('BCS1 BCS3 A318 A319 A320 A321 A20N A21N B733 B734 B735 B736 B737 B738 B739 B38M B752 B753 MD82'.split())
REGIONAL = set('AT45 AT72 AT75 AT76 B190 B462 BE20 CRJ2 CRJ7 CRJ9 CRJX C750 D328 DH8D E135 E145 E170 E190 E195 E290 E75L E75S F100 RJ1H RJ85 SB20 SF34 SW4 AN26'.split())
FEATURES = ['vorige_vertraging', 'gemiddelde_7d', 'vorige_wind', 'vorige_neerslag', 'vorige_temperatuur', 'geplande_bewegingen', 'geplande_landingen', 'widebody_aandeel', 'weekdag', 'maand']
FEATURE_LABELS = {'vorige_vertraging':'Vertraging gisteren','gemiddelde_7d':'Vertraging afgelopen 7 dagen','vorige_wind':'Wind gisteren','vorige_neerslag':'Neerslag gisteren','vorige_temperatuur':'Temperatuur gisteren','geplande_bewegingen':'Roosterdrukte morgen','geplande_landingen':'Landingen morgen','widebody_aandeel':'Aandeel widebody morgen','weekdag':'Dag van de week','maand':'Maand'}


def delay_minutes(planned, actual):
    """Corrigeer alleen duidelijke avond -> volgende vroege ochtend-overgangen.
    Zonder werkelijke datum is een >=6 uur verschil ambigu; apart markeren.
    """
    p = pd.to_timedelta(planned, errors='coerce').dt.total_seconds() / 60
    a = pd.to_timedelta(actual, errors='coerce').dt.total_seconds() / 60
    raw = a - p
    midnight = (p >= 20*60) & (a <= 4*60) & (raw < -12*60)
    corrected = raw.where(~midnight, raw+1440)
    return corrected, midnight, corrected.abs().gt(360)


def load_data():
    raw = pd.read_csv(ROOT/'schedule_airport.csv.gz', na_values=['-'], dtype={'RWY':'string'})
    audit = {'bronrijen':len(raw),'volledig_dubbele_rijen':int(raw.duplicated().sum()),'ontbrekend_per_kolom':raw.isna().sum().to_dict()}
    df = raw.drop_duplicates().copy()
    df['datum'] = pd.to_datetime(df.STD, format='%d/%m/%Y', errors='coerce')
    df['vertraging'], df['middernacht'], df['tijd_ambigu'] = delay_minutes(df.STA_STD_ltc, df.ATA_ATD_ltc)
    df['uur'] = pd.to_timedelta(df.STA_STD_ltc).dt.components.hours
    df['jaar'] = df.datum.dt.year
    df['maand'] = df.datum.dt.month
    df['kwartaal'] = df.datum.dt.quarter
    df['weekdag'] = df.datum.dt.dayofweek
    df['beweging'] = df.LSV.map({'L':'Landing','S':'Vertrek'})
    df['groep'] = np.select([df.ACT.isin(WIDE),df.ACT.isin(NARROW),df.ACT.isin(REGIONAL)],['Widebody','Narrowbody','Regionaal'],default='Onbekend')
    df['geldig'] = df.datum.notna() & df.vertraging.notna() & ~df.tijd_ambigu
    # Negatieve vertraging is vroeg; positieve minuten zijn de voorspeldoelvariabele.
    df['positief'] = df.vertraging.clip(lower=0).where(df.geldig)
    df['te_laat'] = df.vertraging.ge(15).where(df.geldig)
    audit.update(middernachtcorrecties=int(df.middernacht.sum()),ambigue_tijden=int(df.tijd_ambigu.sum()),geldige_vertragingen=int(df.geldig.sum()),onbekende_types=sorted(df.loc[df.groep.eq('Onbekend'),'ACT'].unique().tolist()),gemiddelde_met_ambigue=float(df.vertraging.mean()),gemiddelde_zonder_ambigue=float(df.loc[df.geldig,'vertraging'].mean()))
    cols = ['ID','Name','City','Country','IATA','ICAO','Latitude','Longitude','Altitude','Timezone','DST','TZ','Type','Source']
    airports_raw = pd.read_csv(ROOT/'airports-extended.dat',header=None,names=cols,na_values=[r'\N'])
    # Original OpenFlights file: comma separator + decimal point, unlike lesson clean CSV.
    ap = airports_raw.loc[airports_raw.Type.eq('airport') & airports_raw.ICAO.notna()].copy()
    ap['Latitude'] = pd.to_numeric(ap.Latitude,errors='coerce')
    ap['Longitude'] = pd.to_numeric(ap.Longitude,errors='coerce')
    ap = ap.loc[ap.Latitude.between(-90,90) & ap.Longitude.between(-180,180)]
    audit['dubbele_airport_icao'] = int(ap.ICAO.duplicated().sum())
    # Ambiguous duplicate ICAOs are never silently matched.
    ap = ap.loc[~ap.ICAO.duplicated(keep=False)]
    audit['luchthaven_types'] = airports_raw.Type.value_counts().to_dict()
    df = df.merge(ap[['ICAO','IATA','Name','City','Country','Latitude','Longitude']],left_on='Org/Des',right_on='ICAO',how='left',validate='many_to_one')
    audit['icao_match_vluchten'] = int(df.ICAO.notna().sum())
    audit['icao_match_codes'] = int(df.loc[df.ICAO.notna(),'Org/Des'].nunique())
    audit['bestemmingscodes'] = int(df['Org/Des'].nunique())
    audit['niet_gekoppelde_codes'] = sorted(df.loc[df.ICAO.isna(),'Org/Des'].dropna().unique().tolist())
    weather = pd.concat([pd.read_csv(ROOT/f'weather{y}.csv.gz') for y in [2019,2020]],ignore_index=True)
    weather['datum'] = pd.to_datetime(weather[['year','month','day']])
    if weather.datum.duplicated().any():
        raise ValueError('Weerbron bevat meerdere rijen per dag.')
    weather['tavg'] = weather['temp']
    df = df.merge(weather[['datum','wspd','wpgt','prcp','tavg','pres']],on='datum',how='left',validate='many_to_one')
    audit['weer_dagen'] = len(weather)
    audit['weer_ontbrekend'] = weather[['wspd','wpgt','prcp','tavg','pres']].isna().sum().to_dict()
    audit['rijen_na_joins'] = len(df)
    assert len(df) == len(raw)-audit['volledig_dubbele_rijen']
    # Planned hourly traffic; never actual hour or delay codes in prediction.
    df['gepland_tijdstip'] = df.datum + pd.to_timedelta(df.STA_STD_ltc)
    df['tijdvak'] = df.gepland_tijdstip.dt.floor('h')
    df['roosterdrukte_uur'] = df.groupby('tijdvak')['FLT'].transform('size')
    return df, weather, audit


def daily_frame(df, weather):
    dates = pd.date_range('2019-01-01','2020-12-31',freq='D')
    daily = df.groupby('datum').agg(geplande_bewegingen=('FLT','size'),geplande_landingen=('LSV',lambda x:x.eq('L').sum()),widebody_aandeel=('groep',lambda x:x.eq('Widebody').mean())).reindex(dates)
    # A day absent from source is missing, not automatically zero traffic.
    land = df.loc[df.LSV.eq('L') & df.geldig].groupby('datum').agg(doel=('positief','mean'),landing_n=('FLT','size'))
    daily = daily.join(land).join(weather.set_index('datum')[['wspd','prcp','tavg']])
    daily['vorige_vertraging'] = daily.doel.shift(1)
    daily['gemiddelde_7d'] = daily.doel.shift(1).rolling(7,min_periods=3).mean()
    for orig,new in [('wspd','vorige_wind'),('prcp','vorige_neerslag'),('tavg','vorige_temperatuur')]: daily[new] = daily[orig].shift(1)
    daily['weekdag'] = daily.index.dayofweek
    daily['maand'] = daily.index.month
    daily.index.name = 'datum'
    return daily


def fit_forecast(daily):
    # Fixed split before fitting; test never changes training or interval width.
    usable = daily.loc[daily.doel.notna() & daily.landing_n.ge(10)].copy()
    train = usable.loc[:'2019-08-31']
    cal = usable.loc['2019-09-01':'2019-09-30']
    test = usable.loc['2019-10-01':'2019-12-31']
    stress = usable.loc['2020-01-01':'2020-12-31']
    medians = train[FEATURES].median()
    model = GradientBoostingRegressor(n_estimators=120,max_depth=2,min_samples_leaf=15,learning_rate=.035,loss='huber',random_state=42)
    model.fit(train[FEATURES].fillna(medians),train.doel)
    cp = np.maximum(0,model.predict(cal[FEATURES].fillna(medians)))
    # Empirical 90% absolute residual quantile from separate calibration month.
    width = float(np.quantile(np.abs(cal.doel-cp),.9,method='higher'))
    train_base = float(train.doel.median())
    def evaluate(frame):
        out = frame.copy()
        out['voorspeld'] = np.maximum(0,model.predict(frame[FEATURES].fillna(medians)))
        out['ondergrens'] = (out.voorspeld-width).clip(lower=0)
        out['bovengrens'] = out.voorspeld+width
        out['baseline'] = out.vorige_vertraging.fillna(train_base)
        out['constante_baseline'] = train_base
        out['fout'] = out.doel-out.voorspeld
        metrics = {'dagen':len(out),'MAE':float(mean_absolute_error(out.doel,out.voorspeld)),'RMSE':float(np.sqrt(mean_squared_error(out.doel,out.voorspeld))),'baseline_MAE':float(mean_absolute_error(out.doel,out.baseline)),'constant_MAE':float(mean_absolute_error(out.doel,out.constante_baseline)),'dekking':float(((out.doel>=out.ondergrens)&(out.doel<=out.bovengrens)).mean())}
        return out, metrics
    test_out,metrics = evaluate(test)
    stress_out,stress_metrics = evaluate(stress)
    imp = permutation_importance(model,cal[FEATURES].fillna(medians),cal.doel,n_repeats=15,random_state=42,scoring='neg_mean_absolute_error')
    importance = pd.DataFrame({'kenmerk':[FEATURE_LABELS[x] for x in FEATURES],'MAE_toename':imp.importances_mean,'spreiding':imp.importances_std}).sort_values('MAE_toename',ascending=False)
    return {'test':test_out,'stress':stress_out,'metrics':metrics,'stress_metrics':stress_metrics,'importance':importance,'width':width,'train_n':len(train),'cal_n':len(cal),'train_base':train_base,'imputed_test_cells':int(test[FEATURES].isna().sum().sum())}


def wind_analysis(df, year=2019, threshold=15, trim=False):
    x = df.loc[df.LSV.eq('L') & df.geldig & df.jaar.eq(year) & df.groep.isin(GROUPS) & df.wspd.notna()].copy()
    if trim: x = x.loc[x.vertraging.abs().le(180)]
    x['windgroep'] = np.where(x.wspd.ge(threshold),'Meer wind','Minder wind')
    # Compare daily rates; bootstrap entire days, not individual flights.
    d = x.groupby(['datum','groep','windgroep']).agg(aandeel=('te_laat','mean'),vluchten=('FLT','size')).reset_index()
    rows=[]
    rng=np.random.default_rng(42)
    for (g,w), block in d.groupby(['groep','windgroep']):
        vals=block.aandeel.to_numpy(dtype=float)
        boot=np.mean(rng.choice(vals,size=(1500,len(vals)),replace=True),axis=1)
        rows.append({'groep':g,'windgroep':w,'percentage':100*vals.mean(),'laag':100*np.quantile(boot,.025),'hoog':100*np.quantile(boot,.975),'dagen':len(vals),'vluchten':int(block.vluchten.sum())})
    summary=pd.DataFrame(rows)
    diff=[]
    for group,block in d.groupby('groep'):
        low=block.loc[block.windgroep.eq('Minder wind'),'aandeel'].to_numpy(dtype=float)
        high=block.loc[block.windgroep.eq('Meer wind'),'aandeel'].to_numpy(dtype=float)
        if len(low) and len(high):
            delta=100*(rng.choice(high,(1500,len(high))).mean(axis=1)-rng.choice(low,(1500,len(low))).mean(axis=1))
            diff.append({'groep':group,'verschil_pp':100*(high.mean()-low.mean()),'laag':np.quantile(delta,.025),'hoog':np.quantile(delta,.975)})
    return summary,pd.DataFrame(diff),x[['datum','kwartaal','groep','te_laat','wspd']]
