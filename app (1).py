"""Streamlit-dashboard Zürich, gebouwd rond vragen en onderbouwde bevindingen."""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from analysis import load_data, daily_frame, fit_forecast, wind_analysis, GROUPS

st.set_page_config(page_title='Zürich | Wind, vertraging & corona',page_icon='✈',layout='wide',initial_sidebar_state='expanded')
ORANGE='#E75C37'; TEAL='#087F83'; GREY='#BCC3C8'; DARK='#202C37'; PURPLE='#705CC5'; RED='#BC4053'; BLUE='#3968AA'
PALETTE={'Regionaal':ORANGE,'Narrowbody':GREY,'Widebody':TEAL,'2019':GREY,'2020':ORANGE,'Landing':TEAL,'Vertrek':ORANGE}
st.markdown('''<style>
.stApp {background:#F7F6F2;color:#202C37;}
.block-container {max-width:1250px;padding-top:2.5rem;padding-bottom:3rem;}
h1 {font-weight:750!important;letter-spacing:-1.5px!important;}
h2 {letter-spacing:-.6px!important;} [data-testid="stMetric"] {background:#fff;border-radius:12px;padding:18px;border:1px solid #E2E5E4;}
[data-testid="stMetricValue"] {font-size:2rem;} [data-testid="stSidebar"] {border-right:1px solid #ddd;}
.story {background:#202C37;color:white;padding:24px 28px;border-radius:14px;margin:16px 0 26px;line-height:1.7;font-size:1.08rem;}
.story b {color:#FFA17F;} .eyebrow {color:#64717B;text-transform:uppercase;letter-spacing:2px;font-size:.76rem;margin-bottom:14px;}
</style>''',unsafe_allow_html=True)

@st.cache_data(show_spinner='Brondata inspecteren en koppelen…')
def data(): return load_data()
@st.cache_data(show_spinner='Voorspelling trainen en toetsen op latere dagen…')
def forecast(daily):
    return fit_forecast(daily)
@st.cache_data
def wind(year,threshold,trim,_df):
    return wind_analysis(_df,year,threshold,trim)

def story(text): st.markdown(f'<div class="story">{text}</div>',unsafe_allow_html=True)
def header(kicker,title,subtitle):
    st.markdown(f'<div class="eyebrow">{kicker}</div>',unsafe_allow_html=True)
    st.title(title); st.markdown(subtitle)
def chart(fig,key=None,height=370):
    if fig.layout.title.text:
        st.markdown(f'#### {fig.layout.title.text}')
    fig.update_layout(title=None,height=height,font=dict(family='Arial',size=13,color=DARK),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',margin=dict(l=15,r=25,t=45,b=30),legend=dict(orientation='h',y=1.02,x=0,yanchor='bottom'),hoverlabel=dict(bgcolor='white'),title_font=dict(size=18))
    fig.update_xaxes(showgrid=False,zeroline=False)
    fig.update_yaxes(gridcolor='#E1E4E2',zerolinecolor='#C5CACB')
    st.plotly_chart(fig,use_container_width=True,key=key,config={'displaylogo':False})
def num(n): return f'{int(n):,}'.replace(',','.')
def percent(v): return f'{100*v:.1f}%'.replace('.',',')
def late_by_year(d):
    return d.loc[d.geldig].groupby('jaar').agg(n=('FLT','size'),late=('te_laat','mean'),positive=('positief','mean'))
def monthly_counts(d):
    x=d.groupby(['jaar','maand','beweging']).size().rename('vluchten').reset_index()
    full=pd.MultiIndex.from_product([[2019,2020],range(1,13),['Landing','Vertrek']],names=['jaar','maand','beweging'])
    # Missing cells stay NaN: no automatic zero days/months for absent source rows.
    return x.set_index(['jaar','maand','beweging']).reindex(full).reset_index()

D,W,A=data()
AIRPORTS=D[['Org/Des','IATA','City']].drop_duplicates().set_index('Org/Des')
def airport_code(code):
    value=AIRPORTS.IATA.get(code)
    return str(value) if pd.notna(value) and len(str(value))==3 else 'code onbekend'
def airport_label(code):
    city=AIRPORTS.City.get(code)
    return f'{airport_code(code)} · {city if pd.notna(city) else "luchthavennaam onbekend"}'
with st.sidebar:
    st.markdown('### ZÜRICH / ZRH')
    st.caption('Van rooster naar inzicht · 2019–2020')
    PAGE=st.radio('Verhaal', ['Vertraging & voorspelling','Wind & vliegtuigtype','Het coronajaar'],label_visibility='collapsed')
    st.divider()
    st.caption('Lees per grafiek de titel, kleurlegenda en korte toelichting.')
    st.caption('Vertraagd: ≥15 minuten. Landing en vertrek worden waar nodig apart geanalyseerd.')
    st.caption('Bronnen: HvA-rooster, Meteostat 06670 en OpenFlights. Alle bronbestanden zijn meegeleverd.')

if PAGE=='Vertraging & voorspelling':
    header('01 / begrijpen → toetsen','Vertraging herkennen én voorspellen','Eerst het patroon per uur; daarna een eerlijke toets: kunnen we morgen de gemiddelde aankomstvertraging voorspellen?')
    year=2019
    st.caption('Analyse van 2019. De vergelijking met 2020 staat op de coronapagina.')
    x=D.loc[D.jaar.eq(year)&D.geldig]
    h=x.groupby(['uur','beweging']).agg(percentage=('te_laat','mean'),n=('FLT','size')).reset_index()
    reliable=h.loc[h.n.ge(100)]
    if not reliable.empty:
        peak=reliable.loc[reliable.percentage.idxmax()]
        story(f'In {year} heeft <b>{peak.beweging.lower()} rond {int(peak.uur):02d}:00</b> het hoogste aandeel vertraagde vluchten onder uurvakken met minstens 100 bewegingen: <b>{percent(peak.percentage)}</b>. Dat is een patroon in het rooster, geen bewezen effect van het tijdstip.')
    fig=go.Figure()
    for b,col in [('Landing',TEAL),('Vertrek',ORANGE)]:
        hh=h.loc[h.beweging.eq(b)].set_index('uur').reindex(range(24))
        fig.add_trace(go.Scatter(x=hh.index,y=100*hh.percentage,name=b,mode='lines+markers',line=dict(color=col,width=3),customdata=hh.n,hovertemplate='%{x}:00 · %{y:.1f}%<br>n=%{customdata}<extra>%{fullData.name}</extra>',connectgaps=False))
    fig.update_layout(title=f'{peak.beweging} rond {int(peak.uur):02d}:00 heeft het hoogste vertraagde aandeel · {year}',xaxis_title='Gepland lokaal uur',yaxis_title='Vluchten ≥15 minuten vertraagd (%)')
    chart(fig);st.caption('Alleen geldige tijden; aankomst en vertrek apart. Geen lijn over ontbrekende uren. Het dagritme is geen gecontroleerde causale vergelijking.')
    st.divider();st.subheader('De volgende dag: een getoetste voorspelling')
    F=forecast(daily_frame(D,W)); M=F['metrics']; T=F['test']
    gain=100*(M['baseline_MAE']-M['MAE'])/M['baseline_MAE']
    direction='lager' if gain>=0 else 'hoger'
    story(f'De gemiddelde fout is <b>{M["MAE"]:.1f} minuten</b> op {M["dagen"]} ongeziene testdagen. Dat is <b>{abs(gain):.1f}% {direction}</b> dan simpelweg de vertraging van gisteren gebruiken. Het model voorspelt een daggemiddelde; de fout voor een individuele vlucht kan veel groter zijn.')
    c=st.columns(3)
    c[0].metric('Gemiddelde voorspellingsfout',f'{M["MAE"]:.1f} min');c[1].metric('Fout als we gisteren herhalen',f'{M["baseline_MAE"]:.1f} min');c[2].metric('Testdagen binnen de foutband',percent(M['dekking']))
    st.caption('Een kleinere fout is beter. De foutband toont een verwachte marge; de dekking vertelt op hoeveel testdagen de echte vertraging binnen die marge viel.')
    t=T
    plot_t=t.reindex(pd.date_range(t.index.min(),t.index.max(),freq='D'))
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=plot_t.index,y=plot_t.bovengrens,line=dict(width=0),showlegend=False,hoverinfo='skip',connectgaps=False))
    fig.add_trace(go.Scatter(x=plot_t.index,y=plot_t.ondergrens,line=dict(width=0),fill='tonexty',fillcolor='rgba(112,92,197,.13)',name='Empirische 90%-foutband',hoverinfo='skip',connectgaps=False))
    fig.add_trace(go.Scatter(x=plot_t.index,y=plot_t.doel,name='Werkelijk',line=dict(color=DARK,width=1.8),connectgaps=False))
    fig.add_trace(go.Scatter(x=plot_t.index,y=plot_t.voorspeld,name='Voorspeld',line=dict(color=PURPLE,width=2.5),connectgaps=False))
    fig.update_layout(title='In 2019 voorspelt het model beter dan gisteren herhalen',xaxis_title='Voorspelde dag',yaxis_title='Gemiddelde positieve aankomstvertraging (min)')
    chart(fig,height=420)
    st.caption('Train: jan–aug 2019. Foutband: absolute voorspelfouten in september 2019 (90e percentiel). Test: okt–dec 2019. De band is empirisch en biedt geen gegarandeerde dekking bij veranderende omstandigheden.')
    st.markdown('**Hoe voorspellen we morgen?**')
    st.write('Het model gebruikt vertraging en weer van eerdere dagen, plus het geplande verkeer, de weekdag en maand van morgen. Het ziet geen werkelijke vertraging of gemeten weer van morgen. Eerst trainen we op januari–augustus 2019, daarna testen we op latere dagen.')
    st.caption('Geannuleerde vluchten ontbreken mogelijk in het rooster. Dit is daarom een historische toets; voor dagelijks gebruik is een volledig vooraf gepubliceerd rooster nodig. Ontbrekende invoer wordt ingevuld met een typische waarde uit de trainingsperiode.')
    l,r=st.columns([1,1])
    with l:
        imp=F['importance'].head(6).sort_values('MAE_toename')
        cols=[ORANGE if v==imp.MAE_toename.max() and v>0 else GREY for v in imp.MAE_toename]
        fig=go.Figure(go.Bar(x=imp.MAE_toename,y=imp.kenmerk,orientation='h',marker_color=cols,error_x=dict(type='data',array=imp.spreiding,color=DARK)))
        fig.update_layout(title=f'{imp.loc[imp.MAE_toename.idxmax(), "kenmerk"]} helpt het model het meest in september',xaxis_title='Extra MAE na verwisselen (min)',yaxis_title='')
        chart(fig);st.caption('Permutatiebelang op september, geen causaal effect. Balkjes tonen spreiding over 15 verwisselingen. Correlatie tussen kenmerken kan belang verdelen.')
    with r:
        st.markdown('#### Op drukke vertragingsdagen kan het model missen')
        worst=t.loc[t.fout.abs().idxmax()]
        day=t.fout.abs().idxmax().strftime('%d-%m-%Y')
        st.write(f'De grootste misser in deze periode is {day}: voorspeld **{worst.voorspeld:.1f} minuten**, werkelijk **{worst.doel:.1f} minuten** gemiddelde aankomstvertraging.')
        st.caption('De beschikbare gegevens vertellen niet welke storing, staking of weersituatie deze misser veroorzaakte.')
    with st.expander('Bekijk één historische voorspelling'):
        chosen=st.selectbox('Testdag',list(t.index),format_func=lambda v:v.strftime('%d-%m-%Y'))
        row=t.loc[chosen]
        st.write(f'Voorspelling: **{row.voorspeld:.1f} min**. Foutband: **{row.ondergrens:.1f}–{row.bovengrens:.1f} min**. Werkelijk: **{row.doel:.1f} min**. Dit is een historische voorspelling met gisteren bekende informatie, geen voorspelling voor een actuele vlucht.')
    st.download_button('Download alle testvoorspellingen',t.to_csv().encode(),'testvoorspellingen.csv','text/csv')

elif PAGE=='Wind & vliegtuigtype':
    header('02 / de hypothese','Maakt een groter vliegtuig wind minder voelbaar?','Hypothese: op dagen met meer wind neemt vertraging bij kleinere vliegtuigtypes sterker toe dan bij widebody-types.')
    year=2019
    st.caption('Windanalyse van 2019; zo beïnvloedt het afwijkende coronajaar deze vergelijking niet.')
    c=st.columns(2)
    threshold=c[0].slider('Daggemiddelde wind: grens (km/h)',10,25,15)
    trim=c[1].checkbox('Gevoeligheid: zonder |vertraging| >180 min',False)
    summary,diff,x=wind(year,threshold,trim,D)
    rdelta=diff.set_index('groep').verschil_pp
    outcome=f'Regionale types: <b>{rdelta.get("Regionaal",float("nan")):+.1f} procentpunt</b>; widebody-types: <b>{rdelta.get("Widebody",float("nan")):+.1f} procentpunt</b> verschil bij meer wind. '
    story(outcome+f'We vergelijken <b>alleen landingen in {year}</b>. Blauw toont dagen met minder wind; rood toont dagen met meer wind. De afstand tussen beide punten toont het verschil per vliegtuigklasse. “Meer wind” betekent hier een <b>daggemiddelde ≥{threshold} km/h</b> — niet de wind tijdens de landing.')
    st.warning('Grootteklasse is geen gemeten landingsgewicht. Zonder actuele massa, windrichting en uurweer kunnen we niet vaststellen of zwaardere vliegtuigen beter kunnen landen. Uitgevallen of uitgeweken vluchten zijn mogelijk afwezig: ook daardoor kan de groep overblijvende landingen gunstiger lijken.')
    fig=go.Figure()
    for group in GROUPS:
        pair=summary.loc[summary.groep.eq(group)].set_index('windgroep')
        if {'Minder wind','Meer wind'}.issubset(pair.index):
            fig.add_trace(go.Scatter(x=pair.loc[['Minder wind','Meer wind'],'percentage'],y=[group,group],mode='lines',line=dict(color=GREY,width=3),showlegend=False,hoverinfo='skip'))
    for wg,col in [('Minder wind',BLUE),('Meer wind',RED)]:
        b=summary.loc[summary.windgroep.eq(wg)].set_index('groep').reindex(GROUPS)
        fig.add_trace(go.Scatter(y=b.index,x=b.percentage,mode='markers',name=wg,marker=dict(size=13,color=col),error_x=dict(type='data',array=b.hoog-b.percentage,arrayminus=b.percentage-b.laag,color=col),customdata=b[['dagen','vluchten']],hovertemplate='%{y}: %{x:.1f}%<br>%{customdata[0]} dagen · %{customdata[1]} landingen<extra>%{fullData.name}</extra>'))
    wind_direction='sterker' if rdelta.get('Regionaal',0)>rdelta.get('Widebody',0) else 'niet sterker'
    fig.update_layout(title=f'Bij regionale types verandert vertraging {wind_direction} dan bij widebody',xaxis_title='Gemiddeld dagelijks aandeel ≥15 minuten vertraagd (%)',yaxis_title='')
    chart(fig)
    st.caption('Blauw = minder wind; rood = meer wind. De verbinding maakt het verschil binnen één klasse zichtbaar. Elke dag weegt even zwaar. De strepen tonen de onzekerheidsmarge uit 1.500 hersteekproeven van hele dagen; overlap of kleine aantallen maken conclusies minder stevig.')
    for wg in ['Minder wind','Meer wind']:
        rows=summary.loc[summary.windgroep.eq(wg)]
        st.caption(wg + ': ' + ' · '.join(f'{r.groep}: {int(r.dagen)} dagen, {num(r.vluchten)} landingen' for r in rows.itertuples()))
    if not diff.empty:
        regional=diff.loc[diff.groep.eq('Regionaal')]
        heavy=diff.loc[diff.groep.eq('Widebody')]
        if not regional.empty and not heavy.empty:
            a=regional.iloc[0];b=heavy.iloc[0]
            st.markdown(f'**Bij meer wind verandert het vertraagde aandeel met {a.verschil_pp:+.1f} procentpunt bij regionale types en {b.verschil_pp:+.1f} bij widebody-types.**')
            st.write('Dit patroon ondersteunt de richting van de hypothese beschrijvend.' if a.verschil_pp>b.verschil_pp else 'Deze vergelijking ondersteunt de voorgestelde richting van de hypothese niet. Ook dat is een onderzoeksresultaat.')
    st.caption('Regionaal = kleinere jets en turboprops; narrowbody = toestellen met één gangpad; widebody = grotere toestellen met twee gangpaden. Onbekende types tellen hier niet mee. Grootte is een benadering, geen gemeten gewicht.')
    st.subheader('Blijft het windverschil zichtbaar binnen elk kwartaal?')
    st.write('We vergelijken nu meer en minder wind binnen hetzelfde kwartaal. Kies één vliegtuigklasse: elke balk laat één verschil zien, in plaats van zes lijnen tegelijk.')
    group=st.selectbox('Vliegtuigklasse voor de seizoenscontrole',GROUPS,key='season_group')
    daily=x.groupby(['datum','kwartaal','groep']).agg(late=('te_laat','mean'),wind=('wspd','first')).reset_index()
    daily['windgroep']=np.where(daily.wind.ge(threshold),'Meer wind','Minder wind')
    q=daily.loc[daily.groep.eq(group)].groupby(['kwartaal','windgroep']).agg(p=('late','mean'),dagen=('datum','size')).reset_index()
    rates=q.pivot(index='kwartaal',columns='windgroep',values='p').reindex(index=range(1,5),columns=['Minder wind','Meer wind'])
    counts=q.pivot(index='kwartaal',columns='windgroep',values='dagen').reindex(index=range(1,5),columns=['Minder wind','Meer wind']).fillna(0)
    differences=100*(rates['Meer wind']-rates['Minder wind'])
    reliable=counts.min(axis=1).ge(10)&differences.notna()
    shown=differences.where(reliable)
    labels=['Jan–mrt','Apr–jun','Jul–sep','Okt–dec']
    fig=go.Figure(go.Bar(x=labels,y=shown,marker_color=[RED if pd.notna(v) and v>=0 else BLUE for v in shown],text=[f'{v:+.1f}' if pd.notna(v) else '' for v in shown],textposition='outside',customdata=counts[['Meer wind','Minder wind']].to_numpy(),hovertemplate='%{x}: %{y:+.1f} procentpunt<br>Meer wind: %{customdata[0]:.0f} dagen<br>Minder wind: %{customdata[1]:.0f} dagen<extra></extra>',showlegend=False))
    positive=int(shown.gt(0).sum());available=int(shown.notna().sum())
    title=f'{group}: meer wind gaat in {positive} van {available} vergelijkbare kwartalen samen met meer vertraging' if available else f'{group}: te weinig dagen voor een betrouwbare kwartaalvergelijking'
    fig.update_layout(title=title,xaxis_title='Kwartalen van 2019',yaxis_title='Verschil in vertraagd aandeel (procentpunt)')
    fig.add_hline(y=0,line_color=DARK,line_width=1.5)
    for i,v in enumerate(shown):
        if pd.isna(v):
            fig.add_annotation(x=labels[i],y=0,text='Te weinig dagen',showarrow=False,yshift=18,font=dict(color='#64717B',size=11))
    chart(fig,height=380)
    st.caption('Rood boven nul = bij meer wind vaker vertraging. Blauw onder nul = bij meer wind minder vaak vertraging. Bijvoorbeeld +5 betekent 5 procentpunt meer vertraagde landingen, niet 5 minuten vertraging.')
    st.caption('Een balk verschijnt alleen als beide windgroepen minstens 10 dagen bevatten. Elke dag weegt even zwaar. Vergelijken binnen een kwartaal beperkt seizoensverschillen, maar corrigeert niet voor route, maatschappij of drukte en bewijst geen windoorzaak.')
    st.caption(' · '.join(f'{labels[i-1]}: {int(counts.loc[i,"Meer wind"])} dagen met meer wind / {int(counts.loc[i,"Minder wind"])} met minder wind' for i in counts.index))

elif PAGE=='Het coronajaar':
    header('03 / een ander systeem','2020 verandert het verkeer én de vergelijking','Vergelijk dezelfde maanden, dezelfde bewegingen en dezelfde routes. Een gezamenlijk jaargemiddelde verbergt de breuk.')
    tab_trend,tab_map=st.tabs(['Verkeer & vertraging','Europese routekaart'])
    with tab_trend:
        counts=D.groupby('jaar').size();fall=1-counts[2020]/counts[2019]
        story(f'Het geregistreerde verkeer daalt met <b>{percent(fall)}</b>. De volgende vraag is niet alleen hoeveel vluchten verdwijnen, maar ook <b>welke verbindingen overblijven</b> en hoe hun vertraging verandert.')
        movement=st.radio('Beweging',['Beide','Landing','Vertrek'],horizontal=True,key='corona_movement')
        x=D if movement=='Beide' else D.loc[D.beweging.eq(movement)]
        m=monthly_counts(x)
        fig=go.Figure()
        for y in [2019,2020]:
            for b in ['Landing','Vertrek']:
                if movement!='Beide' and b!=movement:continue
                z=m.loc[m.jaar.eq(y)&m.beweging.eq(b)].set_index('maand').reindex(range(1,13))
                fig.add_trace(go.Scatter(x=z.index,y=z.vluchten,name=f'{y} · {b}',mode='lines+markers',line=dict(color=PURPLE if y==2020 else GREY,width=3,dash='solid' if b=='Landing' else 'dash'),connectgaps=False))
        fig.update_layout(title=f'In 2020 zijn er {percent(1-len(x.loc[x.jaar.eq(2020)])/len(x.loc[x.jaar.eq(2019)]))} minder bewegingen',xaxis_title='Maand',yaxis_title='Geregistreerde vluchtbewegingen')
        fig.update_xaxes(dtick=1);chart(fig)
        st.caption('Kleur markeert het jaar; lijnstijl maakt landing/vertrek zichtbaar. Maanden zijn dezelfde noemer, maar hebben verschillende aantallen dagen. De aantallen komen uit het aangeleverde rooster.')
        l,r=st.columns(2)
        with l:
            p=x.loc[x.geldig].groupby(['jaar','maand']).te_laat.mean().mul(100).rename('percentage').reset_index()
            fig=go.Figure()
            for y,col in [(2019,GREY),(2020,TEAL)]:
                z=p.loc[p.jaar.eq(y)].set_index('maand').reindex(range(1,13))
                fig.add_trace(go.Scatter(x=z.index,y=z.percentage,name=str(y),mode='lines+markers',line=dict(color=col,width=3),connectgaps=False))
            fig.update_layout(title=f'Het vertraagde aandeel is in 2020 {"lager" if x.loc[x.geldig & x.jaar.eq(2020)].te_laat.mean()<x.loc[x.geldig & x.jaar.eq(2019)].te_laat.mean() else "hoger"} dan in 2019',xaxis_title='Maand',yaxis_title='≥15 minuten vertraagd (%)');chart(fig)
        with r:
            rt=x.groupby(['Org/Des','jaar']).size().unstack('jaar',fill_value=0).reindex(columns=[2019,2020],fill_value=0)
            top=rt.sort_values(2019,ascending=False).head(10).copy();top['afname']=100*(1-top[2020]/top[2019])
            names=D[['Org/Des','City']].drop_duplicates().set_index('Org/Des').City
            top['label']=[airport_label(i) for i in top.index]
            highlight=top.afname.nlargest(2).index
            colors=[ORANGE if i in highlight else GREY for i in top.index]
            fig=go.Figure(go.Bar(x=top.afname,y=top.label,orientation='h',marker_color=colors,text=top.afname.round(1).astype(str)+'%',textposition='outside'))
            fig.update_layout(title=f'{top.loc[top.afname.idxmax(), "label"]} krimpt het sterkst in de top 10',xaxis_title='Minder bewegingen in 2020 (%)',yaxis_title='');fig.update_yaxes(autorange='reversed');chart(fig,height=430)
            st.caption('Eerst top 10 selecteren op volume 2019, daarna de twee grootste procentuele dalingen markeren. Geen willekeurige keuze van opvallende routes.')
        st.subheader('Vergelijk dezelfde verbindingen')
        route=st.selectbox('Route',list(rt.sort_values(2019,ascending=False).index[:40]),format_func=airport_label)
        z=x.loc[x['Org/Des'].eq(route)&x.geldig].groupby('jaar').agg(bewegingen=('FLT','size'),vertraagd=('te_laat','mean'))
        for yy in [2019,2020]:
            if yy in z.index:
                row=z.loc[yy]
                st.write(f'**{yy}:** {num(row.bewegingen)} bewegingen met een bruikbare tijd; **{percent(row.vertraagd)}** daarvan minstens 15 minuten vertraagd.')
            else:
                st.write(f'**{yy}:** geen bewegingen met bruikbare tijden voor deze selectie.')
        st.caption('Dezelfde route maakt de vergelijking specifieker; veranderingen in tijdstip, toesteltype en maatschappij blijven mogelijke verklaringen. Het rooster bewijst geen afzonderlijk corona-effect op vertraging.')

    with tab_map:
        st.subheader('Welke drukke Europese routes krompen het sterkst?')
        st.caption('Europa is hier operationeel gekozen als luchthavens met een Europe/*-tijdzone in de bron. De selectie is vast op 2019; een verdwijnende route blijft daardoor zichtbaar.')
        ap=pd.read_csv(Path(__file__).parent/'airports-extended.dat',header=None,na_values=[r'\N'])
        europe=set(ap.loc[ap[11].fillna('').str.startswith('Europe/'),5].dropna())
        x=D.loc[D['Org/Des'].isin(europe)&D.ICAO.notna()]
        rt=x.groupby(['Org/Des','jaar']).size().unstack('jaar',fill_value=0).reindex(columns=[2019,2020],fill_value=0)
        top=rt.nlargest(15,2019).copy();top['afname']=100*(1-top[2020]/top[2019])
        top=top.join(D[['Org/Des','City','Name','Latitude','Longitude']].drop_duplicates().set_index('Org/Des'))
        extreme=top.afname.nlargest(2).index
        if len(top):
            winner=top.loc[top.afname.idxmax()]
            story(f'Binnen deze vaste groep krimpt <b>{winner.City}</b> het sterkst: <b>{winner.afname:.1f}%</b> minder bewegingen. De kaart toont <b>15 geselecteerde verbindingen</b>; kleur staat voor procentuele krimp en puntoppervlak voor volume in 2019.')
        # MapLibre open tiles; no Mapbox token. Marker AREA scales with 2019 volume.
        top['code']=[airport_code(i) for i in top.index];top['label']=[f'{row.City} ({airport_code(idx)})' if idx in extreme else '' for idx,row in top.iterrows()]
        fig=go.Figure()
        for _,row in top.iterrows():
            fig.add_trace(go.Scattermap(lon=[8.54917,row.Longitude],lat=[47.46472,row.Latitude],mode='lines',line=dict(width=1,color='#C5CDCF'),showlegend=False,hoverinfo='skip'))
        fig.add_trace(go.Scattermap(lon=top.Longitude,lat=top.Latitude,mode='markers+text',text=top.label,textposition='top right',marker=dict(size=top[2019],sizemode='area',sizeref=2*top[2019].max()/30**2,sizemin=6,color=top.afname,colorscale=[[0,'#FFE6C9'],[.5,'#F89860'],[1,'#B83A20']],cmin=0,cmax=100,showscale=True,colorbar=dict(title=dict(text='Krimp (%)',side='right'),x=1.02,y=.5,len=.8,thickness=14)),customdata=top[['code','City',2019,2020,'afname']],hovertemplate='%{customdata[1]} · %{customdata[0]}<br>2019: %{customdata[2]:,}<br>2020: %{customdata[3]:,}<br>Krimp: %{customdata[4]:.1f}%<extra></extra>',name='Europese verbinding'))
        fig.add_trace(go.Scattermap(lon=[8.54917],lat=[47.46472],mode='markers+text',marker=dict(color=DARK,size=12),text=['Zürich (ZRH)'],textposition='bottom right',name='Zürich',showlegend=False))
        st.markdown(f'#### {winner.City} krimpt het sterkst binnen de 15 drukste Europese routes')
        fig.update_layout(showlegend=False,margin=dict(l=0,r=100,t=10,b=10),paper_bgcolor='rgba(0,0,0,0)',map=dict(style='carto-positron',center=dict(lat=49,lon=9),zoom=2.8),height=560)
        st.plotly_chart(fig,use_container_width=True,config={'displaylogo':False})
        st.caption('Donkerder oranje = sterkere krimp (vaste schaal 0–100%). Groter puntoppervlak = meer bewegingen in 2019; geen radius evenredig aan volume. Alleen de twee grootste krimpers krijgen een tekstlabel. Lijnen zijn schematische verbindingen, geen echte vliegroutes. Achtergrond: CARTO/OpenStreetMap.')
        st.caption(f'{A["bestemmingscodes"]-A["icao_match_codes"]} bestemmingscodes hebben geen betrouwbare luchthavenlocatie in deze bron en krijgen daarom geen kaartpunt.')



st.divider()
st.markdown('**Van bron naar betrouwbare vergelijking**')
st.caption(f'{num(A["bronrijen"])} bronrijen gecontroleerd; {A["middernachtcorrecties"]} duidelijke middernachtovergangen hersteld. {A["ambigue_tijden"]} onduidelijke tijdverschillen tellen mee voor verkeersvolume, maar niet voor vertraging. Koppelingen op luchthaven en datum voegen geen extra vluchten toe.')
st.caption('Dagweer komt van Meteostat-station Zürich-Kloten. Luchthavenlocaties komen van OpenFlights; de vluchtplanning is aangeleverd door HvA. De losse Amsterdam–Barcelona-profielen horen niet bij deze Zürich-vluchten en zijn daarom niet samengevoegd. Verbanden zijn beschrijvend; ze bewijzen geen oorzaak.')
st.markdown('[Weerbron: Meteostat](https://dev.meteostat.net/bulk/daily.html) · [Luchthavenbron: OpenFlights](https://github.com/jpatokal/openflights/blob/master/data/airports-extended.dat)')

st.download_button('Bronnen en methodische toelichting', '- **Rooster:** `schedule_airport.csv` via Brightspace / HvA, door de groep aangeleverd. 323.461 geregistreerde bewegingen, 2019–2020. Het originele CSV-bestand is verliesvrij opnieuw gzip-gecomprimeerd. Publicatierechten van cursusmateriaal blijven bij de oorspronkelijke rechthebbende.\n- **Luchthavens:** [OpenFlights / oorspronkelijke database](https://github.com/jpatokal/openflights/blob/master/data/airports-extended.dat), blob `dc3392c076b769fc28f31fc798cba4af05a5f35b`, opgehaald 5 oktober 2026. 12.668 locaties; 8.264 airports, ook stations/havens/onbekend. [Kaggle-lesbron](https://www.kaggle.com/datasets/open-flights/airports-train-stations-and-ferry-terminals) vereist inloggen. Daarom is de oorspronkelijke database gebruikt, niet het semikolon/decimale-komma lesbestand. Deze bron wordt gelezen met komma en decimale punt; join op **ICAO**, nooit op IATA. [OpenFlights data en ODbL/Database Contents-licentie](https://openflights.org/data.html). De meegeleverde luchthavenbron behoudt die licentie; deze appcode verandert die niet.\n- **Weer:** Meteostat, station **06670 / Zürich-Kloten**, dagbestanden [2019](https://data.meteostat.net/daily/2019/06670.csv.gz) en [2020](https://data.meteostat.net/daily/2020/06670.csv.gz), opgehaald 5 oktober 2026. [Station](https://meteostat.net/en/station/06670), [bestandsformaat](https://dev.meteostat.net/data/timeseries/daily.html), [eenheden](https://dev.meteostat.net/api/stations/daily.html), [licentie](https://dev.meteostat.net/terms.html). Wind km/h, temperatuur °C, neerslag mm, luchtdruk hPa. Bestanden bevatten bronkolommen; Meteostat kan modeldata gebruiken om observatiegaten aan te vullen. De actuele bron vermeldt CC BY-NC 4.0; onderwijsgebruik, geen commerciële verspreiding. De oude link in de opdracht is vervangen door de huidige jaarlijkse endpoint.\n- **Optionele vluchtprofielen:** `flightdata.zip` via HvA, 14 Excel-bestanden voor 7 Amsterdam–Barcelona-vluchten. Bewust niet gekoppeld aan Zürich: andere route en geen betrouwbare joinsleutel. Dit project claimt niet dat vier datasets aan elkaar gekoppeld zijn.\n- **Kaart:** [Plotly Scattermap](https://plotly.com/python/tile-scatter-maps/), MapLibre met CARTO positron / OpenStreetMap; kaartattributie staat op de kaart. De locaties komen uit OpenFlights, niet uit een geschatte positie.\n\n**Code en methoden:** projectcode is voor deze opdracht met hulp van Codex gemaakt en aangepast aan de eigen data; er zijn geen letterlijk overgenomen StackOverflow-snippets. Gebruikte API-documentatie: [Streamlit cache_data](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data), [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing), [pandas merge/validate](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html), [Plotly graph objects](https://plotly.com/python/graph-objects/), [scikit-learn GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html), [MAE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html) en [permutatiebelang](https://scikit-learn.org/stable/modules/permutation_importance.html). De groep moet verwerking, bootstrap, temporele splitsing en MAE zelf kunnen uitleggen.\n', 'SOURCES.md', 'text/markdown')
st.caption('Minor Data Science · Zürich Airport 2019–2020 · historische voorspelling')
