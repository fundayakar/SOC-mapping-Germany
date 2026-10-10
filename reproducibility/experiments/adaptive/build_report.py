"""Build Turkish decision report and scientific figures from completed audit outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def table(d):
    def cell(v):
        if pd.isna(v):return '—'
        if isinstance(v,(float,np.floating)):return f'{v:.3f}'
        return str(v).replace('|','\\|').replace('\n',' ')
    rows=['| '+' | '.join(map(str,d.columns))+' |','| '+' | '.join(['---']*len(d.columns))+' |']
    rows+=['| '+' | '.join(cell(v) for v in row)+' |' for row in d.itertuples(index=False,name=None)]
    return '\n'.join(rows)

def main():
    s=pd.read_csv(HERE/'results/honest_pilot_scores.csv')
    g=pd.read_csv(HERE/'results/honest_gate_pooled_scores.csv')
    summary=pd.read_csv(HERE/'results/gain_summary.csv')
    assoc=pd.read_csv(HERE/'results/feature_associations.csv')
    groups=pd.read_csv(HERE/'results/group_gains.csv')
    stability=pd.read_csv(HERE/'results/year_stability.csv')
    statuses=[json.loads(p.read_text()) for p in (HERE/'pilot').glob('*/completion.json')]
    validation=json.loads((HERE/'validation.json').read_text());assert validation['all_passed']
    pairs=s.pivot(index=['year','target','protocol'],columns='strategy',values='MAE')
    rmse=s.pivot(index=['year','target','protocol'],columns='strategy',values='RMSE')
    lr=s.pivot(index=['year','target','protocol'],columns='strategy',values='log_R2')
    comparisons=[]
    for key in pairs.index:
        for candidate in ['soft_gate_heterogeneity_only','soft_gate_predictions_only','soft_gate_all_with_predictions']:
            for ref in ['fixed_50_target_scale','calibrated_constant_blend','calibrated_constant_physical_MAE','standard_stacking_ridge']:
                comparisons.append(dict(zip(['year','target','protocol'],key),candidate=candidate,reference=ref,
                    delta_MAE=pairs.loc[key,candidate]-pairs.loc[key,ref],delta_RMSE=rmse.loc[key,candidate]-rmse.loc[key,ref],
                    delta_log_R2=lr.loc[key,candidate]-lr.loc[key,ref]))
    comparisons=pd.DataFrame(comparisons);comparisons.to_csv(HERE/'results/pilot_paired_strategy_deltas.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':190})
    fig,axes=plt.subplots(1,2,figsize=(11.7,4.6),constrained_layout=True)
    rows=assoc.query("target=='log' and loss=='logSE'")
    feature_order=['NDVI_std','NDVI_center_mean_abs','NDVI_neighbor_difference','VV_std','VH_std','elev_std','slope_std']
    labels=['NDVI std','NDVI centre–mean','NDVI neighbours','VV std','VH std','Elevation std','Slope std']
    matrix=np.column_stack([rows[(rows.protocol==p)&(rows.year==y)].set_index('feature').loc[feature_order].spearman_rho for p,y in [('block_100',2018),('block_100',2022),('block_100_buffer20',2018),('block_100_buffer20',2022)]])
    axes[0].imshow(matrix,cmap='RdBu_r',vmin=-.3,vmax=.3,aspect='auto')
    axes[0].set_xticks(range(4),['2018\nBlock','2022\nBlock','2018\n+20 km','2022\n+20 km'])
    axes[0].set_yticks(range(7),labels)
    for i in range(7):
        for j in range(4):axes[0].text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=9)
    axes[0].set_title('A. Heterogeneity vs point-level CNN gain\nSpearman ρ; log-trained experts, log squared loss',fontsize=11)
    x=np.arange(4);width=.33
    gg=groups.query("protocol=='block_100' and target=='log' and grouping=='SOC_group'")
    order=['<30','30–<60','60–<150','≥150']
    for i,y in enumerate([2018,2022]):
        a=gg[gg.year.eq(y)].set_index('category').loc[order]
        axes[1].bar(x+(i-.5)*width,a.CNN_RMSE-a.RF_RMSE,width,label=str(y),color=['#267681','#df9a3d'][i])
    axes[1].axhline(0,color='#475569',lw=.8);axes[1].set_xticks(x,['<30\nn=402','30–<60\nn=180','60–<150\nn=61','≥150\nn=20'])
    axes[1].set_ylabel('CNN − contextRF RMSE (g/kg)');axes[1].set_xlabel('Observed SOC group (g/kg), descriptive only')
    axes[1].set_title('B. Overall benefit is not uniform across SOC\nNegative bars favour CNN; log-trained experts',fontsize=11);axes[1].legend(frameon=False)
    fig.savefig(HERE/'SOC_gain_diagnostics.png');plt.close(fig)
    strategies=['fixed_50_target_scale','calibrated_constant_physical_MAE','standard_stacking_ridge','soft_gate_predictions_only','soft_gate_all_with_predictions']
    names=['50:50','MAE\nconstant','Ridge\nstacking','Prediction\ngate','Full\ngate']
    fig,axes=plt.subplots(2,2,figsize=(11.7,7.1),constrained_layout=True)
    for i,(metric,title) in enumerate([('MAE','MAE'),('RMSE','RMSE')]):
        for j,protocol in enumerate(['block_100','block_100_buffer20']):
            ax=axes[i,j];rows=s[(s.target=='raw')&(s.protocol==protocol)]
            for k,y in enumerate([2018,2022]):
                a=rows[rows.year.eq(y)].set_index('strategy')
                delta=a.loc[strategies,metric]-a.loc['calibrated_constant_physical_MAE',metric]
                ax.bar(np.arange(5)+(k-.5)*.34,delta,.34,color=['#267681','#df9a3d'][k],label=str(y))
            ax.axhline(0,color='#475569',lw=.8);ax.set_xticks(range(5),names);ax.set_ylabel(f'Δ {title} (g/kg)')
            ax.set_title(('100 km blocks' if j==0 else '100 km blocks +20 km buffer')+' — '+title,fontsize=11)
            ax.legend(frameon=False)
    fig.suptitle('Raw-trained experts: MAE gains accompany higher RMSE\nRelative to calibration-only MAE-optimal constant blend\nOne outer repeat; negative bars favour the candidate',fontsize=11)
    fig.savefig(HERE/'SOC_honest_gate_pilot.png');plt.close(fig)
    primary=summary.query("protocol=='block_100' and target=='log'")[['year','n','CNN_mean_AE_better_fraction','CNN_AE_wins_all_repeats','RF_AE_wins_all_repeats','mean_gain_AE','high_share_of_net_SE_gain']]
    gate_table=g[g.gate.isin(['heterogeneity_only','predictions_only','all_with_predictions'])][['year','target','protocol','gate','AUC','Brier','constant_Brier']]
    raw_scores=s[(s.target=='raw')&(s.strategy.isin(strategies+['calibrated_constant_blend','contextRF','patchCNN']))][['year','protocol','strategy','RMSE','MAE','log_R2']]
    log_scores=s[(s.target=='log')&(s.strategy.isin(strategies+['calibrated_constant_blend','contextRF','patchCNN']))][['year','protocol','strategy','RMSE','MAE','log_R2']]
    feature_table=assoc.query("protocol=='block_100' and target=='log' and loss=='logSE'").pivot(index='feature',columns='year',values='spearman_rho').reset_index()
    contrasts=comparisons.query("target=='raw' and reference=='calibrated_constant_physical_MAE' and candidate in ['soft_gate_predictions_only','soft_gate_all_with_predictions']")
    path=HERE/'SOC_adaptive_method_pilot_TR.md'
    report=f'''# SOC: CNN kazanım analizi ve uyarlamalı birleştirme pilotu

Tarih: 8 Ekim 2026. Bu rapor tamamlanan deneylerin sonuçlarıdır; yalnızca hazırlanmış kod değildir.

## Karar

**“Görüntü heterojenliğine göre CNN'i seçelim” hipotezi için güçlü bir dayanak bulamadık.** Log hedefte seçim kurallarının ayırt etme gücü zayıf. Ham hedefte ise RF ve CNN'in kendi tahminleri, hangisinin daha düşük hata vereceği hakkında daha fazla bilgi taşıyor. Bu bilgiyle yapılan yumuşak karışımlar MAE/log doğruluğunda yararlı, ancak ham RMSE'de tutarlı üstünlük sağlamıyor.

Bu aşamada yeni bir yöntem bulduğumuz iddiasını kurmuyoruz. **Somut aday, belirli bir hata amacına göre öğrenilen birleştirme; somut sorun ise tipik noktalardaki hata ile yüksek SOC'nin baskın olduğu kare hata arasındaki denge.** Gating ve stacking bilinen yöntem aileleri olduğundan, burada bunları açık baselines olarak uyguluyoruz. Önceki literatür taramasının yenilik sınırları geçerliliğini koruyor.

## İki farklı analiz, iki farklı anlam

1. Mevcut stage-3 held-out tahminlerini eşleştirdik: **663 nokta, 2 predictor yılı, 2 hedef dönüşümü, 2 mekânsal protokol; 21.216 RF–CNN nokta/tekrar eşleşmesi**. Ana protokol 5, tamponlu protokol 3 tekrar. Bu kısım betimleyici/keşif amaçlıdır; bu hatalara bakarak öğrenilen bir kuralı aynı tahminlerle doğrulamıyoruz.
2. Ayrı bir **dürüst üç bölüm pilotu** çalıştırdık: dış katlamalarda sadece önceden sabitlenen **seed=0**, 5 dış fold; 2 yıl ×2 hedef ×2 protokol = **40 RF–CNN uzman çifti**. Her uzman çifti için iç epoch seçimi ve CNN yeniden eğitimi tamamlandı. Beş özellik grubu ile toplam **200 lojistik seçim kuralı** kuruldu; İlk pilotun 13 stratejisine, kalibrasyonda fiziksel MAE'yi doğrudan en aza indiren sabit karışım eklenerek **14 strateji** aynı dış test noktalarında karşılaştırıldı. Dış test tablosu 8 koşulun her birinde 663 nokta içeriyor.

İki yıl da **aynı 2018 SOC hedefini (g/kg, 0–20 cm)** tahmin ediyor. Predictor yılı farkı SOC değişimi değildir. Önceki hatalar bu pilotun sorusunu belirlediği için bu çalışma yeni bir dış veri doğrulaması değildir. Pilotun kendi fit/kalibrasyon/test ayrımları korunmuştur.

## Mevcut deneyde hangi noktalarda CNN kazandırıyor?

Pozitif `gain = loss(contextRF) − loss(CNN)` CNN lehinedir. Her noktada hatalar tekrarlar üzerinden ortalandı; önce tahminleri ortalayıp sonra hata hesaplamadık. Aşağıdaki CNN kazanma oranı, noktanın **ortalama fiziksel mutlak hatası** daha küçükse kazanma sayar.

{table(primary)}

Ana log deneyinde CNN noktaların yaklaşık yarısında kazanıyor. 2018'de 127, 2022'de 128 noktada beş tekrarın hepsinde mutlak hata daha düşük; birçok noktada kazanan tekrarlar arasında değişiyor. Ortalama MAE farkı küçük: yaklaşık **0,061 ve 0,187 g/kg** CNN lehine.

Yüksek SOC grubunun net kare hata kazancına payı 2018'de **2,488**, 2022'de **1,191**. Payın %100'ü aşması hesaplama hatası değildir: ≥150 g/kg grubu iyileşirken geri kalan grubun toplam kare hatası artmıştır. SOC ≥150 g/kg sadece **20 nokta**; eşik mineral/organik toprak sınıflaması değildir.

Aynı noktada 2018/2022 ortalama mutlak hata kazanımlarının Spearman korelasyonu log modeller için **0,362**; kazananın işareti yaklaşık **%60,2** uyumlu. Yıllar aynı hedef ve aynı noktaları kullandığı için bağımsız tekrar olarak yorumlanamaz.

### Heterojenlik ilişkisi

Görüntüden NDVI/SAR/topografya kanal standart sapmaları, merkez–çevre ortalama farkı ve komşu NDVI piksel farkı üretildi. Bunlar gerçek SOC bilgisi kullanmıyor. Aşağıdaki tablo log hedefli uzmanlarda nokta bazındaki log kare hata kazanımıyla Spearman ilişkisini gösterir. Çoklu keşif taraması olduğu için p değerleri veya “anlamlı keşif” iddiası vermiyoruz.

{table(feature_table)}

Örneğin NDVI standart sapması için 2018/2022 korelasyonları **0,039/0,053**: belirgin bir tek değişkenli seçim kuralı göstermiyor. Tek bir değişkenle zayıf ilişki olması bütün doğrusal olmayan yöntemlerin başarısız olacağı anlamına gelmez; bu yüzden ayrıca dürüst pilot çalıştırıldı.

Gözlenen 2018 LUCAS arazi örtüsü sadece betimleyici alt grup tablosunda kullanıldı. Cropland grubunda 276 noktada log CNN'in RMSE'si her iki predictor yılında daha düşük; diğer gruplardaki sonuçlar aynı değil. Çok küçük gruplardan yorum çıkarılmadı. **Saha arazi örtüsü etiketini deploy edilebilir bir görüntü değişkeni gibi seçim modeline sokmadık.** Bu alt grup örüntüsü nedensel açıklama değildir.

## Dürüst pilot nasıl kuruldu?

Her dış eğitim kümesi 100 km bloklarıyla 3-fold GroupKFold'un ilk ayrımından bölündü:

- **A — uzman eğitimi:** RF/CNN yalnızca bu noktalardan öğrenir; burada {min(x['n_expert'] for x in statuses)}–{max(x['n_expert'] for x in statuses)} nokta.
- **B — kalibrasyon:** aynı dondurulmuş uzmanların bu bölümdeki hatalarından seçim kuralı, sabit ağırlık ve stacking öğrenilir; {min(x['n_calibration'] for x in statuses)}–{max(x['n_calibration'] for x in statuses)} nokta.
- **C — dış test:** uzman ve seçim kuralının birlikte sonucunu ölçer; hedefleri hiçbir fit veya ağırlık seçimine girmez.

CNN epoch sayısı A'nın içindeki ayrı mekânsal eğitim/validasyon ayrımında seçildi, sonra yalnızca A üzerinde yeniden eğitildi. **Seçim kuralı kurulduktan sonra uzmanlar A+B üzerinde yeniden fit edilmedi**; kalibrasyon ve test tahminleri aynı uzmanlara aittir. Bu, uzmanların veri miktarını azaltır ama dürüst bir ilk pilot sağlar. Stage-3 uzmanları tüm dış eğitim kümesini kullanıyordu; onların sayısal skorlarıyla bu pilotun skorlarını doğrudan yöntem farkı olarak karşılaştırmamalıyız.

Tamponlu protokolde A–B, A–C, B–C ve iç epoch eğitim–validasyon çiftlerinin tamamı 20 km'den uzaktır. Standart protokolde de blok paylaşımı ve patch örtüşmesi yoktur. Tampon veri sayısını ve eğitim kapsamını da değiştirir; protokoller arası farkı yalnızca mesafe etkisine atfetmiyoruz. Girdi/hedef normalizasyonu uzman eğitiminde; gate StandardScaler yalnızca B üzerinde fit edilir.

**Seçim kuralı:** B'de hangi uzmanın hedef ölçeğinde daha küçük mutlak hata verdiğini lojistik regresyonla tahmin ettik. L2 düzenleme C=0,1, sabit; tuning yok. Ağırlık, tahmin edilen CNN kazanma olasılığıdır. Beş özellik seti: merkez+iklim 17; heterojenlik 29; uzman çıktıları ve farkları 3; merkez/iklim+heterojenlik 46; hepsi 49. Gerçek SOC, koordinat, saha arazi örtüsü ve test hatası özellik değildir.

**Karşılaştırmalar:** contextRF; CNN; hedef ölçeğinde 50:50; fiziksel ölçekte 50:50; B'de hedef MSE'sini en aza indiren tek sabit konveks ağırlık; B'de fiziksel MAE'yi en aza indiren tek sabit konveks ağırlık; B'de daha iyi görünen tek uzman; iki uzman çıktısıyla StandardScaler+Ridge(alpha=10) stacking; beş yumuşak gate; tüm özelliklerle sert seçim. MAE ağırlığı analitik ağırlıklı medyanla bulunur; yalnızca B üzerinde 1.001 ağırlıklı grid ile kontrol edildi. B'nin gerçek hedeflerinin kullanılması kalibrasyonun amacıdır; C'nin hedefleri burada kullanılmaz.

Log deneyindeki yumuşak karışım log1p tahminlerini birleştirir ve expm1 ile döner; ham deney fiziksel tahminleri birleştirir. Bu iki işlem aynı değildir. Tahminlerde negatif fiziksel değerler 0'a sınırlandı; SOC'ye göre clipping/tuning yapılmadı.

## Kural kazananı ayırt edebiliyor mu?

AUC=0,5 sıralama açısından ayırt etme yok demektir; AUC kazanım büyüklüğünü ölçmez. Brier düşükse daha iyidir. `constant_Brier`, her dış fold'un sadece B'deki CNN kazanma oranını tüm C noktalarına veren sabit olasılık referansıdır. Pooled AUC fold'lara ait skorları birleştirir; fold sonuçları da ayrı dosyada korunur.

{table(gate_table)}

Log hedefte AUC'ler yaklaşık **0,49–0,55**; gate Brier skorları sabit olasılık referansını iyileştirmiyor. Ham hedefte **yalnızca uzman tahminlerini kullanan** kuralın AUC'si dört yıl/protokol koşulunda yaklaşık **0,649 /0,674 /0,681 /0,683**. Tüm görüntü özelliklerini eklemek bu sinyali tutarlı artırmıyor. Heterojenlik tek başına ham hedefte yaklaşık **0,543–0,595** AUC aralığında.

Bu, “yüksek heterojenlikte CNN'e geç” hikâyesini desteklemekten çok, **uzman çıktılarının hata davranışı** üzerine çalışmayı destekler. AUC artışı nihai SOC RMSE/MAE kazancı anlamına gelmez.

## SOC tahmininde ne değişti?

### Ham hedefli uzmanlar

{table(raw_scores)}

Kalibrasyonda fiziksel MAE'ye göre seçilen sabit karışıma eşleştirilmiş farklar; negatif RMSE/MAE ve pozitif log R² aday lehine:

{table(contrasts[['year','protocol','candidate','delta_RMSE','delta_MAE','delta_log_R2']])}

Tüm özellikli yumuşak gate MAE'ye göre ayarlanmış sabit karışıma göre **MAE'yi dört koşulun hepsinde 0,32–1,35 g/kg azaltıyor**, log R² de dört koşulda yükseliyor. **RMSE dört koşulun hepsinde 0,05–0,59 g/kg artıyor.** MSE'ye göre ayarlanmış sabit karışımla karşılaştırıldığında MAE farkı 1,45–1,74 g/kg; bu daha büyük farkın bir kısmı optimizasyon amacı farkından kaynaklanıyor. Bu nedenle güçlü MAE referansına karşı sonuç ana yorumdur. Yalnızca tahminlere dayanan daha basit gate de MAE/log R²'yi dört koşulda iyileştiriyor; tüm görüntü özelliklerine her durumda ihtiyaç görünmüyor.

Sabit 50:50, kalibrasyonda seçilmiş sabit karışımdan farklı kayıplarda daha iyi olabiliyor. Standart stacking 2018 tamponlu ham deneyde MAE bakımından tüm özellikli gate'den daha iyi. Baselines'ın başarılarını saklamıyoruz. Bu sonuçların **tek dış tekrar** olması nedeniyle birinci çıkan varyantı doğrulanmış nihai yöntem olarak seçmiyoruz.

### Log hedefli uzmanlar

{table(log_scores)}

Bazı karışımlar tek modellerden daha iyi; bu, kazanan sınıflandırmasının güçlü olduğu anlamına gelmez. Hataları farklı iki modelin sabit karışımı da varyansı azaltabilir. Log hedefte gate için güvenilir kazanan ayırt etme sinyali bulunmadığından, küçük skor farklarından mekanizma iddiası kurmuyoruz.

## Yöntem geliştirme için ne yapacağız?

**Hipotezi daraltıyoruz:** görüntü heterojenliğinin kazananı belirlediği varsayımı yerine, uzmanlar arasındaki farkın ve seçilen hata amacının birleştirmeyi nasıl etkilediğini sınayacağız. Bu pilot yeni bir algoritma adı veya yenilik kanıtı üretmez.

Buradaki gate **P(CNN kazanır)** öğreniyor. İki modelden birini beklenen kaybı küçültmek için seçerken asıl gereken **E[loss(RF)−loss(CNN) | mevcut girdiler]** değerinin işaretidir. Bir model çok sayıda noktada küçük miktarda kazanıp birkaç noktada çok büyük kaybedebilir; kazanma olasılığı bu büyüklük farkını dikkate almaz. Yumuşak olasılık ağırlığı da hedef MSE'sini otomatik en aza indirmez. Bu yüzden sonraki adayın ağırlığı doğrudan açıkça seçilmiş kayba göre öğrenmesi gerekir; önce bunun bilinen koşullu ensemble yaklaşımlarından farkını literatürde sınamalıyız.

İlk takip deneyi için iki öncelik var:

1. **CNN'in yararlı bilgi kaynağını ayırmak:** aynı CNN'i merkez pikseli bütün patch'e yayarak ve merkez korunmuş çok bantlı piksel vektörlerini mekânsal olarak permüte ederek ayrı ayrı eğitmek. Böylece ağ kapasitesi, çevre dağılımı ve uzamsal düzenin katkılarını daha iyi ayırırız. CNN'in özgün uzamsal düzen kullandığı gösterilmeden bu katkı üzerine yeni yöntem kurmayalım.
2. **Kayıp amaçlı birleştirme:** basit prediction-only gate/50:50/stacking'i güçlü referans olarak koruyup, ağırlığı kalibrasyon kaybına doğrudan göre öğrenen aday geliştirmek. Sonraki doğrulamada dış eğitimin içinde mekânsal cross-fit uzman tahminleri gerekir; bütün mevcut OOF hatalarını kullanıp aynı OOF üzerinde başarı ölçülmez. Dış test tekrarlarını artırmak ve dış coğrafyada sınamak nihai iddia için gerekli.

Tekrarlar arası SD güven aralığı değildir. 20 yüksek SOC noktasına dayanarak güvenilir tail sınıflandırması, organik/mineral rejim ayrımı veya evrensel yüksek SOC yöntemi iddiası kurmuyoruz. Bütün olumlu/olumsuz varyantlar CSV dosyalarında korunuyor.

## Kontroller ve dosyalar

- 21.216 mevcut eşleştirmede nokta/yıl/hedef/fold eşitliği, kaynak hedef ve tekil kimlik kontrol edildi.
- 40 uzman çiftinin kayıtlı RF/CNN tahminleri yeniden üretildi; normalizasyon yalnızca A'dan hesaplanmış.
- 200 gate için scaler ortalama/varyansının yalnızca B'den geldiği kontrol edildi; sınıflandırıcılar sadece B'nin yeniden üretilmiş uzman hatalarıyla tekrar fit edildi ve aynı katsayılar elde edildi. Kayıtlı dış tahminler de tekrar üretildi.
- Test etiketlerini değiştirmenin CNN tahminlerini değiştirmediği çalıştırılarak doğrulandı.
- Tamponlu bütün rol çiftlerinde ölçülen minimum uzaklık **{validation['min_buffer_role_distance_m']:.3f} m**. Ayrımlar, kimlikler, mesafeler, epoch eğrileri ve ağırlıklar pakette.
- Başarılı kontrol kaydı `validation.json`; tam sonuçlar `results/`; modeller/katlamalar `pilot/`; kaynak SHA256 ve tasarım `source_manifest.json` /`pilot_design.json`.

Yeniden çalışma: önceki `SOC_stage3_package.zip` stage-3 paketini açın ve Python/PyTorch ortamını kurun. `python analyze_gains.py --stage3-root /path/to/stage3`; `python honest_pilot.py --stage3-root /path/to/stage3 --workers 4`; `python validate_pilot.py --stage3-root /path/to/stage3`; `python add_mae_reference.py`; `python build_report.py`. Eğitim scripti aynı tasarımlı tamamlanmış fold'ları tekrar kullanır. Kod veya tasarım değişirse yeni çıktı klasörü gerekir.

Paket bütün yeni analiz kodlarını, çıkarılmış özellikleri, eşleştirilmiş stage-3 tahminlerini, pilot kalibrasyon/test tahminlerini, gate/stacking ve CNN ağırlıklarını içerir. Büyük RF ara checkpoint'leri pakete eklenmedi; RF'ler kaynak veri ve sabit ayarlarla yeniden üretilebilir; RF checkpoint'i yoksa `validate_pilot.py` sadece ilgili RF'yi kayıtlı A noktaları üzerinde deterministik olarak yeniden fit eder. İlk stage-3 rasterları bu eki şişirmemek için tekrar eklenmedi; kaynak paket kimliği ve dosya hash'leri manifestte.

Bu sonuç raporu önceki literatür taramasıyla birlikte değerlendirilmelidir; yeni bir sistematik tarama veya algoritmanın literatürde bulunmadığına dair iddia içermez.
'''
    path.write_text(report,encoding='utf-8')
    (HERE/'README.md').write_text('''# SOC adaptive-method exploratory pilot

Read `SOC_adaptive_method_pilot_TR.md` first. This is a completed exploratory study, not a claim of a novel algorithm.

Requirements: Python 3.12, numpy, pandas, scipy, scikit-learn, matplotlib, joblib, PyTorch CPU (used 2.14.1+cpu).

The original stage-3 package supplies real patches and unchanged expert code. Commands:

```bash
python analyze_gains.py --stage3-root /absolute/path/to/stage3
python honest_pilot.py --stage3-root /absolute/path/to/stage3 --workers 4
python validate_pilot.py --stage3-root /absolute/path/to/stage3
python add_mae_reference.py
python build_report.py
```

All 40 expert-pair jobs and 14 strategies (including an additional MAE-optimized constant reference) have completed. Outputs retain negative results. Source hashes, spatial roles, all calibration/test predictions, 200 gates, 40 CNN weights and report figures are included. Large intermediate RF checkpoints are excluded; the validation script deterministically refits any missing RF on the recorded expert-only cohort. Do not bypass fingerprint guards after modifying the design.

Saved neural-normalization arrays `mean`, `sd`, `cmean`, `csd` must be loaded as float32, matching their original raster/climate statistics. Scalar `ymean`, `ysd` are Python floats. `validate_pilot.py` demonstrates correct replay.

Do not fit a selector to global existing OOF errors and evaluate its SOC performance on those same errors. The honest pilot fits experts on A, gate on disjoint B, and evaluates on C. Experts stay frozen after gate training. One outer repeat and a smaller expert-training cohort constrain inference.
''',encoding='utf-8')
    print('Saved report and figures',path)

if __name__=='__main__':main()
