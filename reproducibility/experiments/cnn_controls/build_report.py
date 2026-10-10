"""Turkish report and scientific figures for completed context controls/loss gates."""
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
        return str(v).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(map(str,d.columns))+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(cell(v) for v in row)+' |' for row in d.itertuples(index=False,name=None)])

def main():
    v=json.loads((HERE/'validation.json').read_text());gv=json.loads((HERE/'loss_gate/validation.json').read_text());assert v['all_passed'] and gv['all_passed']
    s=pd.read_csv(HERE/'results/scores_by_repeat.csv');epochs=pd.read_csv(HERE/'results/epochs.csv')
    main_variants=['center_tiled','shuffle_314159','original_fresh']
    primary=s[s.variant.isin(main_variants)]
    summary=primary.groupby(['protocol','year','target','variant']).agg(repeats=('seed','size'),RMSE=('raw_rmse','mean'),RMSE_SD=('raw_rmse','std'),
        MAE=('raw_mae','mean'),log_R2=('log_r2','mean'),log_R2_SD=('log_r2','std'),raw_R2=('raw_r2','mean'),low_RMSE=('below150_raw_rmse','mean'),high_bias=('ge150_raw_bias','mean')).reset_index()
    summary.to_csv(HERE/'results/ablation_summary.csv',index=False)
    paired=[]
    for (protocol,year,target),g in primary.groupby(['protocol','year','target']):
        for reference in ['center_tiled','shuffle_314159']:
            aa=g[g.variant.eq('original_fresh')].set_index('seed');bb=g[g.variant.eq(reference)].set_index('seed').loc[aa.index]
            dr=aa.raw_rmse-bb.raw_rmse;dm=aa.raw_mae-bb.raw_mae;dl=aa.log_r2-bb.log_r2
            paired.append(dict(protocol=protocol,year=year,target=target,comparison='original−'+reference,repeats=len(aa),
                delta_RMSE=dr.mean(),RMSE_improved=int((dr<0).sum()),delta_MAE=dm.mean(),delta_log_R2=dl.mean(),log_R2_improved=int((dl>0).sum()),
                min_delta_log_R2=dl.min(),max_delta_log_R2=dl.max()))
    paired=pd.DataFrame(paired);paired.to_csv(HERE/'results/paired_ablation_effects.csv',index=False)
    secondary=s[s.seed.eq(0)&s.variant.isin(['original_fresh','shuffle_314159','shuffle_271828'])][['protocol','year','target','variant','raw_rmse','raw_mae','log_r2']]
    secondary.to_csv(HERE/'results/permutation_sensitivity_seed0.csv',index=False)
    matched=primary[primary.seed.lt(3)].groupby(['protocol','year','target','variant'])[['raw_rmse','raw_mae','log_r2']].mean().reset_index()
    matched.to_csv(HERE/'results/matched_first_three_repeats.csv',index=False)
    l=pd.read_csv(HERE/'loss_gate/scores.csv');delta=[]
    for (year,protocol),g in l.groupby(['year','protocol']):
        g=g.set_index('variant')
        for objective in ['MAE','MSE']:
            candidate='direct_'+objective
            refs=['calibrated_constant_physical_MAE' if objective=='MAE' else 'calibrated_constant_blend','winner_tuned_'+objective]
            for ref in refs:
                delta.append(dict(year=year,protocol=protocol,candidate=candidate,reference=ref,
                    delta_RMSE=g.loc[candidate,'raw_rmse']-g.loc[ref,'raw_rmse'],delta_MAE=g.loc[candidate,'raw_mae']-g.loc[ref,'raw_mae'],
                    delta_log_R2=g.loc[candidate,'log_r2']-g.loc[ref,'log_r2']))
    delta=pd.DataFrame(delta);delta.to_csv(HERE/'loss_gate/paired_deltas.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':185})
    fig,axes=plt.subplots(2,2,figsize=(11.7,7.8),constrained_layout=True)
    names=['Centre tiled','Pixels shuffled','Original layout'];colors=['#267681','#df9a3d']
    for i,metric in enumerate(['log_R2','RMSE']):
        for j,protocol in enumerate(['block_100','block_100_buffer20']):
            ax=axes[i,j]
            for k,year in enumerate([2018,2022]):
                g=summary.query('protocol==@protocol and year==@year and target=="log"').set_index('variant').loc[main_variants]
                ax.errorbar(np.arange(3)+(k-.5)*.12,g[metric],yerr=g[metric+'_SD'],fmt='o',capsize=3,color=colors[k],label=str(year))
            ax.set_xticks(range(3),names);ax.set_ylabel('Mean log R²' if metric=='log_R2' else 'Mean physical RMSE (g/kg)')
            ax.set_title(('100 km blocks: 5 repeats' if j==0 else '+20 km buffer: 3 repeats')+' — '+('log R²' if i==0 else 'physical RMSE'),fontsize=11)
            ax.legend(frameon=False)
            ax.set_ylim((.26,.46) if metric=='log_R2' else (44,52))
    fig.suptitle('Same 8,449-parameter CNN: context benefit exceeds the added benefit of spatial ordering\nLog-trained experts; repeat means ±SD (not confidence intervals)',fontsize=12)
    fig.savefig(HERE/'SOC_CNN_information_controls.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.7,4.9),constrained_layout=True)
    cases=[(2018,'block_100'),(2018,'block_100_buffer20'),(2022,'block_100'),(2022,'block_100_buffer20')]
    for j,metric in enumerate(['delta_MAE','delta_RMSE']):
        ax=axes[j]
        for k,obj in enumerate(['MAE','MSE']):
            ref='calibrated_constant_physical_MAE' if obj=='MAE' else 'calibrated_constant_blend'
            d=delta[(delta.candidate=='direct_'+obj)&(delta.reference==ref)].set_index(['year','protocol']).loc[cases]
            ax.bar(np.arange(4)+(k-.5)*.34,d[metric],.34,color=colors[k],label=obj+'-fitted weight')
        ax.axhline(0,color='#475569',lw=.8);ax.set_xticks(range(4),['2018\nBlocks','2018\n+20 km','2022\nBlocks','2022\n+20 km'])
        ax.set_ylabel(('Δ MAE' if j==0 else 'Δ RMSE')+' (g/kg)');ax.legend(frameon=False)
        ax.set_title('Typical-error change' if j==0 else 'Squared-error change',fontsize=11)
    fig.suptitle('Loss-aligned blending vs constants fitted to the same loss\nFrozen raw-target experts; one outer repeat; negative bars favour the candidate',fontsize=12)
    fig.savefig(HERE/'SOC_loss_aligned_blending.png');plt.close(fig)
    rawtable=l[l.variant.isin(['calibrated_constant_physical_MAE','calibrated_constant_blend','direct_MAE','direct_MSE','winner_tuned_MAE','winner_tuned_MSE'])][['year','protocol','variant','raw_rmse','raw_mae','log_r2']]
    e=epochs[epochs.variant.eq('original_fresh')]
    logtable=summary[summary.target.eq('log')][['protocol','year','variant','repeats','RMSE','MAE','log_R2']]
    rawablation=summary[summary.target.eq('raw')][['protocol','year','variant','repeats','RMSE','MAE','log_R2']]
    report=f'''# SOC: aynı CNN ile bilgi kontrolleri ve kayıp amaçlı birleştirme

Tarih: 8 Ekim 2026. Bütün eğitimler ve kayıtlı tahminlerin denetimi tamamlandı.

## Araştırma kararı

**En güçlü bulgu görüntü çevresindeki bilginin katkısı. Özgün mekânsal düzenin ek katkısı daha küçük ve hata ölçeğine bağlı.** Aynı 8.449 parametreli CNN'de merkez bilgisiyle yetinen kontrol geride kalıyor. Çok bantlı piksel dağılımı korunup piksellerin yeri karıştırıldığında bağlam kazancının önemli bir kısmı korunuyor. Log hedefte özgün düzen ortalama log R²'yi biraz artırıyor; ham RMSE'de bu üstünlük tutarlı değil. Ham hedefte karıştırılmış görüntüler birçok koşulda daha iyi.

**Doğrudan kayba göre öğrenilen ağırlık bir aday oluşturdu; standart yöntemlerden açık ve genel bir üstünlük göstermedi.** MAE amaçlı aday, MAE'ye göre ayarlanmış sabit karışımdan dört koşulda daha düşük MAE veriyor. Aynı iç doğrulamayla ayarlanmış kazanan-olasılığı yöntemine karşı MAE'de iki koşulda az daha iyi, ikisinde daha kötü. MSE amaçlı aday sabit MSE karışımına göre dört koşulun ikisinde RMSE'yi çok az azaltıyor. Bu aşamada yeni algoritma veya nihai başarılı yöntem iddiası kurmuyoruz.

Makalenin savunulabilir sorusu: **çevre bilgisinin temsilinin ve eğitim/karışım kaybının, mekânsal genelleme ve farklı SOC hata ölçekleri üzerindeki etkisi**. Yeni bir yöntem hedefi için henüz ek bir mekanizma ve dış doğrulama gerekiyor. Bu sonuçlar hangi fikirlere ağırlık vermemiz gerektiğini daraltıyor; bütün yöntem ailelerinin başarısını veya başarısızlığını kanıtlamıyor.

## Tamamlanan deneyler

- Gerçek **663×14×17×17** patch; iki predictor yılı (2018/2022), iki eğitim hedefi (log1p/ham SOC).
- Ana 100 km bloklarda **5 adet 5-fold**, 20 km tamponlu bloklarda **3 adet 5-fold** tekrar.
- **160 özgün CNN yeniden eğitimi +160 merkez-tiled +160 birinci permütasyon =480 ana model–fold kaydı**.
- İkinci permütasyonla her yıl/hedef/protokolde yalnızca dış seed=0: **40 ek model–fold kaydı**. Toplam **520 CNN model–fold sonucu**, her biri iç epoch seçimi ve dış eğitim kümesine yeniden fit içerir.
- Birleştirme adayı ayrı deney: önceki dürüst A/B/C pilotunun **20 ham hedefli RF–CNN çiftinin sabit tahminleri**. İki kayıp ×iki gate ailesi =**80 son gate**, kalibrasyon içi toplam **720 model fit**; dört koşulda 663'er dış test noktası, **18 strateji**.

Her iki predictor yılı **aynı 2018 SOC ölçümlerini (g/kg, 0–20 cm)** tahmin eder. Bu, SOC değişimi veya bağımsız zamansal doğrulama değildir. Ana/tamponlu tekrar ortalamaları karşılaştırılırken 5/3 sayısı korunur; ilk üç eşleştirilmiş tekrar ayrıca CSV'de verilir. Tampon, eğitim sayısını ve coğrafi kapsamı da azaltır; protokol farkını sadece mesafe etkisi olarak okumuyoruz. Tekrar SD'si güven aralığı değildir.

## Aynı CNN'de neyi değiştirdik?

| Girdi | Çevre değerleri | Mekânsal düzen | Merkez/iklim dalı | Parametre |
| --- | --- | --- | --- | --- |
| original_fresh | Gerçek 14 bant ×17×17 | Gerçek | Gerçek merkez14 +iklim3 | 8.449 |
| center_tiled | Bütün pikseller orijinal merkez vektörü | Sabit görüntü | Aynı | 8.449 |
| shuffle_314159 | Aynı çok bantlı piksel vektörleri | Nokta içinde karıştırılmış; merkez korunur | Aynı | 8.449 |
| shuffle_271828 | Aynı; ikinci bağımsız düzen | Aynı işlem, ikinci seed | Aynı | 8.449 |

Her noktada bütün 14 bant **birlikte** taşındı; bantlar ayrı ayrı karıştırılmadı. Noktalar arasında piksel alışverişi yok. Merkez (8,8) yerinde kaldı; bütün piksel değerleri ve çok değişkenli vektörleri birebir korundu. Dönüşüm tersine uygulanarak orijinal rasterların aynen geri geldiği doğrulandı. Permütasyon point_id ve önceden seçilmiş seed'in hash'inden gelir; SOC etiketi, fold, koordinat veya test performansı kullanılmaz. Aynı noktada iki yıl ve tekrarlar için aynı permütasyon uygulanır.

**Karıştırılmış model hem eğitim hem testte karıştırılmış görüntülerle ayrı eğitildi.** Testte tek başına görüntü bozup eğitim dağılımını değiştirme yaklaşımı kullanılmadı. Bu örneklemde bütün maskeler 1; geçerlilik/missingness yapısı değişmedi. Dönüşüm piksel komşuluklarını değiştirir; bu bir pedolojik müdahale veya arazi süreci için nedensel deney değildir.

Girdi normalizasyonu bütün varyantlarda **orijinal, dönüştürülmemiş ilgili eğitim patch'lerinden** hesaplandı. Böylece merkez/iklim dalına gelen normalize edilmiş 17 değer birebir aynı kaldı. İç normalizasyon yalnızca iç eğitimden, dış normalizasyon yalnızca dış eğitimden gelir. Aynı ağ, Adam lr=0,001, wd=0,0001, batch32, max100epoch, patience15; aynı başlangıç/loader seed'i `42+100×outer_seed+fold`. Her girdi varyantı kendi iç doğrulamasında epoch seçti. Bu, belirli eğitim prosedürü altındaki kontrollü girdi karşılaştırmasıdır; tüm mimarilerin optimum performansı iddiası değildir.

### Özgün CNN neden yeniden eğitildi?

Yeni fit fonksiyonu ile mevcut kaynak fit fonksiyonu aynı örnek koşuda **epoch/eğri/tahminleri birebir** üretti. Eski kayıtların eğitimini bit düzeyinde yeniden üretmek ise bütün koşularda gerçekleşmedi. 160 özgün koşunun **{int((e.epochs!=e.old_original_epochs).sum())}** tanesinde seçilen epoch farklı; eski/yeni tahminler arasında en büyük nokta farkı **{e.old_original_max_prediction_delta.max():.3f} g/kg**. İlk kontrolün çok küçük farkı, tüm eğitim koşularının farkını temsil etmez; erken durdurma seçiminin hassasiyeti önemlidir. Farkın bütün teknik nedenlerini ayrı ayrı izole etmiş değiliz.

Bu nedenle ana karşılaştırma, kontrol modelleriyle **aynı güncel fit kodunda yeniden eğitilmiş original_fresh** kullanır. Eski CNN tahminleri sadece tarihsel karşılaştırma olarak korunur. Yeni 520 ağırlığın tamamı tekrar yüklenip kayıtlı tahminleri **birebir (maksimum fark {v['max_prediction_replay_delta']:.1e})** verdi. Kaydedilmiş raster/iklim normalizasyon vektörleri float32, hedef ortalama/SD skalarları Python float olarak yeniden yüklenmeli; tip dönüşümü inference aritmetiğini değiştirebilir.

## Log hedefli CNN: bağlam ve düzen

{table(logtable)}

Eşleşmiş tekrar farkları; RMSE/MAE negatifse, log R² pozitifse özgün düzen lehine:

{table(paired[paired.target.eq('log')][['protocol','year','comparison','repeats','delta_RMSE','RMSE_improved','delta_MAE','delta_log_R2','log_R2_improved']])}

Özgün CNN'nin merkez-tiled'a göre ortalama log R² artışı ana protokolde **2018:0,0867 /2022:0,0870**, tamponlu protokolde **0,0656 /0,0590**. Bu, merkez MLP karşılaştırmasına ek olarak **aynı kapasitede bağlam bilgisinin katkısını** destekler.

Özgün düzenin birinci permütasyona göre ortalama log R² artışı **0,0126–0,0268**. Ana protokolde iki yılda da 4/5; tamponluda 2018 3/3, 2022 2/3 tekrar log R² açısından iyileşiyor. **Fiziksel RMSE'de dört yıl/protokol ortalamasının yalnızca biri özgün düzen lehine:** 2018 ana koşulda yaklaşık 0,27 g/kg iyileşme; diğer üçünde karıştırılmış modelin RMSE'si daha düşük. Bir metriğe göre üstünlük diğer metriğe taşınmıyor.

## Ham hedefli CNN

{table(rawablation)}

Karıştırılmış CNN ham hedefte dört koşulun tamamında özgün CNN'den daha düşük ortalama MAE ve daha yüksek ortalama log R² veriyor. Ortalama ham RMSE'de 2018 ana koşul özgün düzen lehine; diğer üçü karıştırma lehine. 2022 tamponlu ham koşulda özgün CNN log R²'si merkez-tiled'dan da daha düşük. Ham hedef eğitimi, yüksek değerlerin kare hata ağırlığı ile tipik nokta/log doğruluğu arasında farklı bir denge yaratıyor; “CNN her koşulda daha iyi” yorumu desteklenmiyor.

## İkinci karıştırma seed'i

{table(secondary)}

Bu tablo sadece aynı dış seed=0'ı karşılaştırır; 5/3 tekrar ortalamasıyla karıştırılmaz. Log hedefte 2022'nin ana ve tamponlu koşullarında özgün düzen her iki permütasyondan daha yüksek log R² verir. 2018'de küçük farkların işareti permütasyona göre değişebilir. Bu yüzden ikinci seed'i bütün tekrarlar için güçlü sağlamlık doğrulaması olarak sunmuyoruz; ilk duyarlılık kontrolüdür.

## Ağırlığı doğrudan kayba göre öğrenmek

Bu bölüm yeni ablation ağlarını birleştirmez. **Önceki dürüst pilotun A'da eğitilmiş, B/C'de tahmin üretmiş aynı sabit RF–CNN uzman çiftlerini kullanır.** Uzmanların eğitim kümesi ablation deneyindeki tam dış eğitimden daha küçüktür. Bu iki bölümün mutlak skorlarını birbirine karşı model üstünlüğü olarak okumamalıyız.

Birleştirme:

`w(x) = sigmoid(b + beta · z(x))`

`SOC_hat(x) = RF_hat(x) + w(x) × (CNN_hat(x) − RF_hat(x))`

`z` sadece RF tahmini, CNN tahmini ve farklarının kalibrasyon eğitiminde standardize edilmiş değerleri. Gerçek SOC veya arazi örtüsü deploy girdisi değildir. Seçilen kalibrasyon kaybı +slope L2 düzenleme minimize edilir. MAE, 0,01 g/kg düzleştirmeli mutlak hatayla yaklaşık optimize edilir; iç doğrulama gerçek MAE'yi kullanır. MSE gerçek fiziksel kare hatadır. Kayıp normalizasyon ölçeği ilgili gate eğitimindeki SOC MAD'ıdır (minimum1). İntercept cezasız; ağırlık 0–1, nihai tahmin iki uzman arasında.

Lambda grid **0,01/0,1/1**; her B içinde 100 km bloklarla 3-fold, tamponlu koşulda iç ayrımda da 20 km mesafe. Her adayın scaler'ı iç eğitimden gelir. Pooled iç held-out MAE/MSE ile lambda seçilir; son fit yalnızca B'de yapılır. Üç önceden belirlenmiş başlangıçla L-BFGS-B; bütün 40 son doğrudan-loss fit'i solver convergence bildirdi. Global optimum garantisi yoktur.

Güçlü karşılaştırma: aynı girdileri kullanan kazanan-olasılığı lojistik gate için **C=0,1/1/10** aynı iç fold'larda ve aynı nihai MAE/MSE ölçütüyle seçildi. Ayrıca MAE/MSE'ye göre ayarlanmış sabit ağırlıklar, 50:50 ve önceki stacking/gate varyantları tutuldu. Farklı regularizasyon grid'leri ve sınırlı katsayılar nedeniyle karşılaştırmayı saf kayıp etkisinin tek başına izolasyonu diye sunmuyoruz. Uzmanlar A+B üzerinde sonradan yeniden eğitilmedi. Dış test C hedefi ağırlık/hyperparametre seçiminde kullanılmadı.

{table(rawtable)}

Eşleşmiş farklar:

{table(delta)}

**MAE adayı:** MAE-optimal sabit karışıma göre MAE dört koşulda yaklaşık **0,57 /1,73 /1,06 /1,37 g/kg** daha düşük. Ham RMSE üç koşulda daha yüksek, 2022 tamponluda yaklaşık **0,21 g/kg** daha düşük. Aynı iç seçimle güçlendirilmiş winner gate'e karşı MAE'de iki koşulda iyileşme yalnızca **0,004 /0,056 g/kg**; diğer ikisinde yaklaşık **0,076 /0,075 g/kg** daha yüksek. Bu, tutarlı yeni yöntem üstünlüğü değildir.

**MSE adayı:** MSE-optimal sabit karışıma göre RMSE 2018'de yaklaşık **0,197 /0,062 g/kg** daha yüksek; 2022'de **0,063 /0,055 g/kg** daha düşük. MAE dört koşulda yaklaşık **0,89–1,25 g/kg** azalıyor. Farklar tek dış tekrar ve aynı 663 noktaya dayanır; dört koşul bağımsız dört veri seti değildir. Çok küçük RMSE farklarını güçlü ilerleme diye sunmuyoruz.

## Bilimsel sınırlar ve bir sonraki yöntem kararı

- Merkez kontrolü ve çok bantlı karıştırma, **komşu değerlerin bilgisi ile gerçek düzenin bilgisini** daha iyi ayırdı. Bağlamı öğrenmek için mutlaka gerçek piksel komşuluğuna ihtiyaç olduğu gösterilmedi; birleştirme amacı ve metriğe göre sonuç değişiyor.
- Sigmoid karışım, lojistik seçim ve doğrudan-loss ağırlık bilinen yöntem araçlarıdır. Bu prototipin yapılmış olması tek başına algoritmik yenilik değildir. Önceki literatür taramasındaki sınırlar sürüyor.
- SOC ≥150 g/kg **20 gözlem**; mineral/organik etiket değildir. Yüksek SOC'ye özel güçlü model veya evrensel tail rejimi iddiası kurmuyoruz. Ham/log ve alt grup hataları CSV'lerde birlikte korunur.
- Önceki dış hatalar bu takip sorularını belirledi. Eğitim/test ayrımlarının dürüst olması, bütün keşif sürecini yeni bir bağımsız dış doğrulama haline getirmez. Ülke dışı veya yeni ölçüm kohortu gerekir.
- Bir sonraki aday için **çevredeki değer dağılımını daha verimli temsil etme ve açıkça seçilmiş hata amacına göre birleştirme** üzerine odaklanabiliriz. Böyle bir mimari/prosedürün literatürdeki yakın karşılıklarını kontrol etmeden yeni isim veya yenilik iddiası kurmayacağız. Mevcut gate varyantlarında genel üstünlük görülmediği için onları nihai yöntem diye seçmiyoruz.

## Denetim ve yeniden çalışma

`validation.json`: 520 ağırlık replay'i; merkez/iklim dalında birebir eşitlik; eğitime özgü orijinal normalizasyon; fold/rol kimlikleri; tampon mesafeleri; etiket değiştirmenin inference'ı değiştirmemesi. Minimum ölçülen tamponlu iç/dış mesafe **{v['minimum_buffer_inner_or_outer_distance_m']:.3f} m**. `loss_gate/validation.json`: 80 son gate replay'i, B'ye özgü scaler ve seçilen hyperparametreler, 720 iç fit; MAE/MSE analitik türevleri sonlu farkla kontrol edildi.

Ana giriş noktası **`complete_experiment.py`**; `run_ablations.py` ortak dönüşüm/fit çekirdeği ve korunmuş ilk tasarım sürümüdür. Çekirdeğin eski kayıtlara sıkı replay tanısını doğrudan çalıştırmak yerine tamamlanmış eşleşmiş deney girişini kullanın. Değişen bilimsel tasarım/kod ayrı klasör gerektirir; fingerprint kontrolünü kaldırmayın.

Paket `stage3_reference/` altında kullanılan gerçek NPZ'leri, fold dosyasını ve kaynak eğitim kodlarını; `prior_reference/` altında birleştirme için sabit uzman tahminlerini içerir. Önceki uzman eğitimlerinin tüm ağırlıkları yerine bu sabit girdiler ve kaynak kimliği verilir; önceki tam paket ayrıca korunmuştur. Yeni 520 CNN ve80gate ağırlığı, bütün tahminler/roller, permütasyon eşlemeleri, hata/epoch tabloları ve şekiller bu pakettedir.

Paket klasöründe:

```bash
python complete_experiment.py --stage3-root ./stage3_reference --workers 4
python loss_aligned_gate.py --stage3-root ./stage3_reference --prior-root ./prior_reference
python validate_experiment.py --stage3-root ./stage3_reference
python build_report.py
```

Python3.12, PyTorch CPU2.14.1, numpy/pandas/scipy/scikit-learn/matplotlib/joblib; sürümler ve hash'ler `environment.json` içinde. Kontrol kayıtları tamamsa aynı tasarımda tekrar kullanılır. Yeni platformda eski ağırlıkların inference'ı çok küçük aritmetik farklar verebilir; yeni deney karşılaştırması aynı platformda birlikte üretilmelidir.
'''
    (HERE/'SOC_CNN_controls_and_loss_gate_TR.md').write_text(report,encoding='utf-8')
    (HERE/'README.md').write_text('''# SOC context controls and loss-aligned blending

Read SOC_CNN_controls_and_loss_gate_TR.md. Completed: 520 new CNN model-fold fits and 80 final conditional gates; all stored predictions audited.

Use Python 3.12 and the recorded environment in environment.json (torch 2.14.1+cpu). From this folder:

```bash
python complete_experiment.py --stage3-root ./stage3_reference --workers 4
python loss_aligned_gate.py --stage3-root ./stage3_reference --prior-root ./prior_reference
python validate_experiment.py --stage3-root ./stage3_reference
python build_report.py
```

complete_experiment.py is the main entry. run_ablations.py is the pinned shared transform/fit core and initial diagnostic version; do not invoke its old strict historical-replay driver as the completed experiment.

stage3_reference supplies real patches, source folds, source training code and historical predictions. prior_reference supplies frozen expert calibration/test predictions for the separate loss-gate pilot. Full earlier expert training packages are retained separately. No RF intermediate forests are duplicated here.

All three primary CNN inputs use exactly the same network, point branch, common original-training normalization, initialization and folds. Each separately retrained variant chooses epochs only within outer training. Shuffle moves multiband vectors jointly within each point, keeps the center, and is applied to both train and test.

Saved normalization vectors must retain float32; ymean/ysd scalars must be Python float. validate_experiment.py shows correct loading. A Python float and a float64 numpy zero-dimensional array can change float32 inverse-normalization arithmetic.

The conditional gate uses existing raw-target experts from disjoint A/B/C. Its hyperparameters are selected only within B. The mixture is a standard sigmoid linear gate optimized for a specified loss; this is not a new-algorithm claim. Negative and mixed outcomes are retained.

Scientific changes require a fresh output folder. Never remove fingerprint guards. For full new training on another platform, use a fresh copy without completed run checkpoints and preserve the published copy.
''',encoding='utf-8')
    print('Saved report and figures',flush=True)

if __name__=='__main__':main()
