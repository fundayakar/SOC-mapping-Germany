"""Build the report only from completed RF target and real-patch experiments."""
from pathlib import Path
import json,hashlib,zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
ART=ROOT/'deliverables';ART.mkdir(exist_ok=True)

def table(df):
    df=df.copy()
    for c in df:
        if pd.api.types.is_float_dtype(df[c]):df[c]=df[c].map(lambda v:f'{v:.3f}' if pd.notna(v) else '')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in df.itertuples(index=False,name=None))

def cnn_summary():
    pooled=[];pred=[];training=[]
    for year in [2018,2022]:
        folder=ROOT/'cnn/results'/str(year);status=json.loads((folder/'completion.json').read_text())
        assert status['all_completed'] and status['cnn_trained']
        pooled.append(pd.read_csv(folder/'pooled_oof_by_repeat.csv'))
        pred.append(pd.read_csv(folder/'oof_predictions.csv',dtype={'point_id':str}))
        for p in folder.glob('*.json'):
            r=json.loads(p.read_text())
            if 'point_id' in r:
                training.append({k:r[k] for k in ['model','target','year','protocol','seed','fold','selected_epochs','elapsed_seconds']})
    pooled=pd.concat(pooled,ignore_index=True);pred=pd.concat(pred,ignore_index=True);training=pd.DataFrame(training)
    assert len(training)==400 and len(pred)==53040 and len(pooled)==80
    assert pooled.n.eq(663).all() and pred.groupby(['model','target','year','seed']).point_id.nunique().eq(663).all()
    keys=['model','target','year'];metrics=['raw_r2','log_r2','raw_rmse','raw_mae','raw_bias','below150_raw_mae','below150_raw_rmse','ge150_raw_bias','ge150_sse_fraction']
    summary=pooled.groupby(keys)[metrics].agg(['mean','std','min','max']).reset_index()
    summary.columns=['_'.join(c).rstrip('_') for c in summary.columns]
    summary.to_csv(ROOT/'cnn/performance_summary.csv',index=False)
    training.to_csv(ROOT/'cnn/training_diagnostics.csv',index=False)
    changes=[]
    for model,reference in [('contextRF','centerRF'),('centerMLP','centerRF'),('patchCNN','centerRF'),('patchCNN','contextRF'),('patchCNN','centerMLP')]:
        k=['target','year','seed'];p=pooled[pooled.model.eq(model)].merge(pooled[pooled.model.eq(reference)],on=k,suffixes=('_new','_reference'),validate='one_to_one')
        for _,r in p.iterrows():
            row={c:r[c] for c in k};row['model']=model;row['reference']=reference
            for metric in metrics:row['delta_'+metric]=r[metric+'_new']-r[metric+'_reference']
            changes.append(row)
    changes=pd.DataFrame(changes);changes.to_csv(ROOT/'cnn/paired_changes_vs_centerRF.csv',index=False)
    effects=changes.groupby(['model','reference','target','year']).agg(delta_raw_r2=('delta_raw_r2','mean'),delta_log_r2=('delta_log_r2','mean'),delta_rmse=('delta_raw_rmse','mean'),delta_rmse_min=('delta_raw_rmse','min'),delta_rmse_max=('delta_raw_rmse','max'),RMSE_improved_repeats=('delta_raw_rmse',lambda x:int((x<0).sum()))).reset_index()
    effects.to_csv(ROOT/'cnn/paired_changes_summary.csv',index=False)
    old=pd.read_csv(ROOT/'stage2/results/oof_predictions.csv',dtype={'point_id':str}).query("cohort=='global_0_20' and protocol=='block_100' and model=='M6'")
    baseline=pred[pred.model.eq('centerRF')&pred.target.eq('log')].merge(old,on=['year','protocol','seed','fold','point_id'],validate='one_to_one')
    assert len(baseline)==663*5*2
    parity=float(abs(baseline.pred_SOC-np.expm1(baseline.pred_log)).max())
    # No assumption of bitwise identity after float32 serialization; report the actual difference.
    assert parity<1e-5,'Center RF changed materially from the stage-2 M6 reference'
    validation=json.loads((ROOT/'cnn/data/patch_validation.json').read_text())
    return pooled,pred,summary,training,effects,parity,validation

def buffered_summary():
    scores=[];fits=[]
    for year in [2018,2022]:
        folder=ROOT/'cnn/results_buffer20'/str(year)
        status=json.loads((folder/'completion.json').read_text());assert status['all_completed'] and status['cnn_trained']
        scores.append(pd.read_csv(folder/'pooled_oof_by_repeat.csv'))
        for p in folder.glob('*.json'):
            r=json.loads(p.read_text())
            if 'point_id' in r:fits.append(r)
    assert len(fits)==240
    p=pd.concat(scores,ignore_index=True);assert len(p)==48 and p.n.eq(663).all()
    metrics=['raw_r2','log_r2','raw_rmse','raw_mae','raw_bias','ge150_raw_bias','below150_raw_rmse']
    s=p.groupby(['model','target','year'])[metrics].agg(['mean','std','min','max']).reset_index();s.columns=['_'.join(c).rstrip('_') for c in s.columns]
    s.to_csv(ROOT/'cnn/buffer20_performance_summary.csv',index=False)
    changes=[]
    for model,reference in [('contextRF','centerRF'),('patchCNN','centerRF'),('patchCNN','contextRF')]:
        keys=['target','year','seed'];q=p[p.model.eq(model)].merge(p[p.model.eq(reference)],on=keys,suffixes=('_new','_reference'),validate='one_to_one')
        for _,r in q.iterrows():
            changes.append(dict(model=model,reference=reference,target=r.target,year=int(r.year),seed=int(r.seed),delta_log_r2=r.log_r2_new-r.log_r2_reference,delta_raw_r2=r.raw_r2_new-r.raw_r2_reference,delta_rmse=r.raw_rmse_new-r.raw_rmse_reference))
    c=pd.DataFrame(changes);c.to_csv(ROOT/'cnn/buffer20_paired_changes.csv',index=False)
    e=c.groupby(['model','reference','target','year']).agg(delta_log_r2=('delta_log_r2','mean'),delta_raw_r2=('delta_raw_r2','mean'),delta_rmse=('delta_rmse','mean'),min_delta_rmse=('delta_rmse','min'),max_delta_rmse=('delta_rmse','max'),RMSE_improved_repeats=('delta_rmse',lambda x:int((x<0).sum()))).reset_index()
    e.to_csv(ROOT/'cnn/buffer20_paired_changes_summary.csv',index=False)
    return s,e

def figures(targets,cnn,buffer):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8))
    methods=['log_naive','log_spatial_smear','raw_SOC_RF'];colors=['#23719e','#679e88','#dd8743']
    m=targets.query("protocol=='block_100' and model=='M6'")
    for ax,metric,title in zip(axes,['raw_rmse','raw_mae'],['Raw SOC RMSE (g/kg)','Raw SOC MAE (g/kg)']):
        for i,method in enumerate(methods):
            q=m[m.method.eq(method)].set_index('year').loc[[2018,2022]]
            x=np.arange(2)+(i-1)*.24;ax.bar(x,q[metric+'_mean'],width=.22,color=colors[i],label=['Log / naive','Log / spatial smearing','Raw SOC target'][i]);ax.errorbar(x,q[metric+'_mean'],yerr=q[metric+'_std'],fmt='none',ecolor='#333',capsize=3,lw=1)
        ax.set_xticks([0,1],['2018 predictors','2022 predictors']);ax.set_ylabel(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(loc='lower left',framealpha=1,fontsize=9)
    fig.suptitle('Target-scale diagnostic: M6, matched 100 km spatial folds',fontsize=13)
    fig.text(.5,.015,'663 points. Bars: mean of five repetitions. Error bars: repeat SD, not confidence intervals.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.94));fig.savefig(ART/'SOC_target_transform_comparison.png',bbox_inches='tight');plt.close(fig)
    models=['centerRF','contextRF','centerMLP','patchCNN'];names=['Center RF','Context RF','Center MLP','Patch CNN'];x=np.arange(4)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8))
    for ax,target in zip(axes,['log','raw']):
        for year,offset,color in [(2018,-.17,'#23719e'),(2022,.17,'#dd8743')]:
            q=cnn[cnn.target.eq(target)&cnn.year.eq(year)].set_index('model').loc[models]
            ax.bar(x+offset,q.raw_r2_mean,width=.32,color=color,label=str(year));ax.errorbar(x+offset,q.raw_r2_mean,yerr=q.raw_r2_std,fmt='none',ecolor='#333',capsize=3,lw=1)
        ax.set_xticks(x,names,rotation=12);ax.axhline(0,color='#666',lw=.8);ax.set_ylabel('Pooled out-of-fold raw SOC R²');ax.set_title('Training target: '+('log(1 + SOC)' if target=='log' else 'raw SOC'));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(title='Predictor year',loc='best',framealpha=1)
    fig.suptitle('Real image patches: RF / MLP / CNN controls, 100 km spatial folds',fontsize=13)
    fig.text(.5,.015,'663 identical points; five 5-fold repetitions. Error bars: repeat SD.\nCNN uses 17×17 patches; three climate covariates enter as a separate vector.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.13,1,.94));fig.savefig(ART/'SOC_real_patch_model_comparison.png',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8),sharey=True)
    for ax,year in zip(axes,[2018,2022]):
        for source,name,color in [(cnn,'100 km, 5 repetitions','#23719e'),(buffer,'100 km +20 km buffer, 3 repetitions','#dd8743')]:
            q=source[source.target.eq('log')&source.year.eq(year)].set_index('model').loc[models]
            ax.errorbar(x,q.log_r2_mean,yerr=q.log_r2_std,marker='o',capsize=3,label=name,color=color,lw=1.5)
        ax.set_xticks(x,names,rotation=12);ax.set_title(str(year)+' predictors');ax.set_ylim(0,.5);ax.grid(alpha=.2);ax.set_ylabel('Pooled out-of-fold log SOC R²')
    axes[0].legend(loc='lower left',fontsize=9,framealpha=1)
    fig.suptitle('Spatial sensitivity of image-context models: log training target',fontsize=13)
    fig.text(.5,.015,'663 matched points; error bars are repeat SD. Buffered protocol removes training points within 20 km of test.\nTargets in both predictor-year experiments are the same 2018 measurements.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.13,1,.94));fig.savefig(ART/'SOC_patch_buffer_sensitivity.png',bbox_inches='tight');plt.close(fig)

def report(targets,pooled,cnn,training,effects,parity,validation,buffer,buffer_effects):
    m6=targets.query("protocol=='block_100' and model=='M6'")
    ctable=cnn[['year','target','model','raw_r2_mean','raw_rmse_mean','raw_mae_mean','log_r2_mean','ge150_raw_bias_mean']]
    ttable=m6[['year','method','raw_r2_mean','raw_rmse_mean','raw_mae_mean','raw_bias_mean','log_r2_mean','ge150_raw_bias_mean']]
    mt=targets.query("protocol=='block_100'")
    best_target=mt.sort_values('raw_rmse_mean').iloc[0]
    best_cnn=cnn.sort_values('raw_rmse_mean').iloc[0]
    neural=training[training.model.isin(['centerMLP','patchCNN'])]
    years=[]
    for year in [2018,2022]:
        for target in ['log','raw']:
            q=cnn[cnn.year.eq(year)&cnn.target.eq(target)].set_index('model')
            e=effects[effects.year.eq(year)&effects.target.eq(target)&effects.model.eq('patchCNN')&effects.reference.eq('centerRF')].iloc[0]
            years.append(f"- {year}, {target} hedef: CNN ham R² **{q.loc['patchCNN','raw_r2_mean']:.3f}**, merkez RF **{q.loc['centerRF','raw_r2_mean']:.3f}**, çevre RF **{q.loc['contextRF','raw_r2_mean']:.3f}**. CNN − merkez RF RMSE farkı **{e.delta_rmse:+.2f} g/kg**; RMSE {int(e.RMSE_improved_repeats)}/5 tekrarda iyileşti.")
    models=pd.DataFrame([
        dict(model='centerRF',inputs='14 merkez raster değeri + 3 iklim skaları',capacity='800 ağaç'),
        dict(model='contextRF',inputs='Merkez + kanal başına çevre mean/std/min/max + 3 iklim + geçerlilik oranı',capacity='800 ağaç'),
        dict(model='centerMLP',inputs='14 merkez raster değeri + 3 iklim skaları',capacity='1.121 parametre'),
        dict(model='patchCNN',inputs='14 kanal ×17×17 + maske; merkez değerleri ve 3 iklim skaları ayrı dalda',capacity='8.449 parametre')])
    report=f'''# SOC — hedef dönüşümü ve gerçek görüntü parçalarıyla RF–MLP–CNN

## Tamamlanan işler

**1.550 RF eğitimi** hedef ölçeği/geri dönüşüm deneyi için tamamlandı. Ardından gerçek GEE rasterlarından iki yılın **663×14×17×17** görüntü parçaları üretildi. PyTorch ile dört model, iki hedef ölçeği, iki predictor yılı değerlendirildi: ana 100 km doğrulamada beş adet 5-fold tekrar (**400 model–fold sonucu**), 20 km tamponlu doğrulamada üç adet 5-fold tekrar (**240 model–fold sonucu**). Görüntü bağlamı deneyinin toplamı **640 model–fold sonucu**; ağların her biri için iç doğrulama eğitimi ve tüm dış eğitim kümesine yeniden eğitim yapıldı.

Her iki yıldaki hedef aynı **2018 SOC konsantrasyonu (g/kg), 0–20 cm** ölçümleridir. Predictor yılı karşılaştırması SOC değişimi veya bağımsız zamansal doğrulama değildir. Ana dış katlamalar önceki aşamayla aynıdır; her tekrarın tüm test tahminleri birleştirilip pooled skor hesaplandı. Tekrar SD'si bir güven aralığı değildir. Bu, önceki test hatalarından hareketle tasarlanmış yöntem tanı deneyidir; yeni dış veri üzerinde bağımsız doğrulama olarak sunulmamalı.

## Geri dönüşüm düzeltmesi ne değiştirdi?

Üç yöntem karşılaştırıldı: log1p hedef + doğrudan expm1 geri dönüşümü; log1p hedef + eğitim içi mekânsal cross-fit smearing; ham SOC hedefinde RF. Her dış eğitim kümesinin içinde 100 km bloklarla üç iç fold kullanıldı. İç fold'da held-out log hata `e = log1p(SOC) − tahmin` üzerinden `S = mean(exp(e))` hesaplandı; dış test tahmini `max(0, S × exp(tahmin_log) − 1)` oldu. Dış test hedefleri katsayı hesaplamasına girmedi. Tamponlu protokolde iç eğitimde de 20 km uzaklık koşulu uygulandı.

{table(ttable)}

M6–2018'de ham R² **0,089 → 0,113**, RMSE **49,34 → 48,69 g/kg**; buna karşılık MAE **19,36 → 21,42 g/kg** ve log R² **0,353 → 0,260**. Genel bias −7,87'den −0,70 g/kg'ye yaklaşsa da yüksek SOC grubunda ortalama hata −231,4'ten yalnızca −220,7 g/kg'ye değişti. Ham hedef RF'sinde yüksek SOC hatası daha az negatif olsa da genel MAE arttı. Bu yüzden genel bias'ın sıfıra yaklaşmasını yüksek SOC tahminlerinin düzeldiği şeklinde yorumlamıyoruz.

Beş değişken grubunun tümünde ve iki yılda smearing ham RMSE'yi ortalamada düşürdü; artışın büyüklüğü sınırlı. En düşük ortalama ham RMSE bu tanı deneyinde **{best_target.model}, {int(best_target.year)}, {best_target.method}: {best_target.raw_rmse_mean:.2f} g/kg** (ham R² {best_target.raw_r2_mean:.3f}). Bu seçimi testlere göre ayarlanmış ve bağımsız doğrulanmış nihai üretim modeli olarak sunmuyoruz; bütün varyantlar tablolarda korunuyor.

Tek bir global smearing katsayısı, hata dağılımının her çevresel koşulda aynı olduğunu garanti etmez. İç modeller dış modelden daha az eğitim verisiyle kurulur; bu fark da kalibrasyonu etkileyebilir. Bu uygulama Duan'ın geri dönüşüm fikrinin mekânsal cross-fit RF tanı uyarlamasıdır; orijinal makalenin bütün teorik sonuçlarının bu ağlar/RF için aynen geçerli olduğu iddiası değildir.

## Gerçek görüntü parçaları ve adil kontroller

GEE exportları canlı kontrolden ve tam dosya denetiminden geçti. Her yılda tam 663 kimlik, aynı SOC/derinlik/koordinatlar ve bütün 14 bandın 17×17 şekli doğrulandı. Bu örneklemde bütün patch pikselleri geçerli (minimum geçerli oranı 1,0). Patch merkezleri kaynak noktayı içeren 20 m piksele oturur; en büyük merkez kayması yaklaşık 14,13 m. Raster band sırası ve 3 iklim skalarına ilişkin kontroller kaydedildi.

**Merkez RF'nin log hedef tahminleri önceki M6 referansını yeniden üretiyor:** en büyük mutlak SOC tahmin farkı **{parity:.3g} g/kg**. Böylece yeni model karşılaştırmasının farklı bir nokta/piksel hazırlığına dayandığı kaygısı kontrol edilmiş oldu.

{table(models)}

CNN iki 3×3 convolution katmanı (16/32 kanal), ReLU, global ortalama havuzlama ve merkez/iklim girdilerinin birleştiği küçük bir regresyon başlığı kullanır. Merkez bağlantısı, çevre ortalaması içinde etiketin ait olduğu merkezin kaybolmasını önler. PyTorch sürümü **2.14.1+cpu**; eğitim CPU'da yapıldı. Adam, lr=0,001, weight_decay=0,0001, batch=32, en fazla 100 epoch, patience=15; augmentasyon ve dış test skoruna göre hiperparametre araması yok.

Ağların epoch sayısı dış eğitim kümesindeki önceden belirlenmiş bir iç mekânsal doğrulama ayrımıyla seçildi. Sonra ağ başlangıçtan, tüm dış eğitim kümesi üzerinde seçilen epoch sayısı kadar yeniden eğitildi. Girdi ve hedef normalizasyonu her defasında sadece ilgili eğitim kümesinden hesaplandı. Ham ve log hedefte MSE kullanıldı; fiziksel negatif tahminler 0'a sınırlandı. Eğitim/test görüntü parçalarının örtüşmediği de kontrol edildi.

RF içindeki `contextRF − centerRF` farkı çevre özetlerinin artımlı katkısını sınar. `patchCNN − centerMLP` karşılaştırması hem bağlam bilgisini hem mimari/kapasiteyi değiştirir; **saf mimari etkisi olarak yorumlanamaz**. CNN ile çevre RF de uzamsal bilginin farklı temsillerini kullanır. Bu sınırlar, bir CNN üstünlüğünü yalnızca convolution'a atfetmeyi engeller.

## RF–MLP–CNN sonuçları

{table(ctable)}

{chr(10).join(years)}

Bu dört kontrol içinde en düşük ortalama ham RMSE: **{best_cnn.model}, {int(best_cnn.year)}, {best_cnn.target} hedef — {best_cnn.raw_rmse_mean:.2f} g/kg** (ham R² {best_cnn.raw_r2_mean:.3f}). Model/yıl/hedef karşılaştırmaları aynı noktalar ve aynı katlamalar üzerinde yapıldı. Pozitif/negatif sonuçların tamamı raporlandı; tek başarılı split seçilmedi.

### Eşleşmiş katkılar

RMSE farkında negatif değer `reference` modeline göre iyileşmedir. Min/max beş tekrarın gözlenen aralığı; bir güven aralığı değildir. CNN'nin çevre RF'ye ek kazancı 2018'de log hedefte ortalama yalnızca **0,235 g/kg RMSE** ve 3/5 tekrarda iyileşme; 2022'de **1,394 g/kg** ve 5/5 tekrarda iyileşme. Dolayısıyla merkez RF'ye kıyasla kazancın tamamını convolution mimarisine atfetmiyoruz.

{table(effects)}

### Yüksek SOC hataları

Ana örneklemde SOC ≥150 g/kg olan 20 nokta var. Bu eşik bir mineral/organik toprak etiketi değildir. Tüm modeller bu noktalar eğitimde mevcutken kurulmuştur; aşağıdaki değerler yalnızca aynı held-out tahminlerin betimleyici alt grup hatalarıdır.

{table(cnn[['model','target','year','ge150_raw_bias_mean','ge150_sse_fraction_mean','below150_raw_mae_mean','below150_raw_rmse_mean']])}

Log hedefli CNN'de SOC <150 g/kg grubunun ham RMSE'si 2018/2022 için **19,68/19,98 g/kg**; merkez RF'de **19,34/19,39**, çevre RF'de **18,85/19,32 g/kg**. Dolayısıyla toplam ham kare hatadaki iyileşme bütün SOC aralıklarına yayılmış bir üstünlük değildir; yüksek SOC noktalarındaki iyileşme önemli bir bileşendir. MAE ve log R²'nin yanında bu ayrım da sonuç metninde korunmalı.

Hata ortalaması tek başına yetersizdir: yüksek SOC'yi hâlâ düşük tahmin ederken, düşük SOC'yi artırarak genel bias sıfıra yaklaşabilir. Alt grup tablosu bu dengeyi görünür tutuyor. Yalnızca 20 yüksek SOC noktası bulunduğu için bu grubun sonuçları güçlü sınıflandırma/organik toprak genellemesi olarak sunulmamalı.

## Mekânsal duyarlılık ve sınırlar

Hedef dönüşümü RF deneyinde 100 km +20 km tamponlu test M1 ve M6 ile iki yılda üç tekrar çalıştırıldı. **Görüntü bağlamı kontrolleri de iki yıl, iki hedef ve dört modelle aynı 20 km tamponlu dış katlamalarda üç tekrar tamamlandı.** Dış test ve iç epoch seçimi için kullanılan doğrulama noktalarının 20 km içindeki eğitim noktaları çıkarıldı. Dış testte eğitimle aynı native ERA5-Land hücresini paylaşan nokta bulunmuyor; raster parçalarının örtüşmediği de kontrol edildi. Tamponlu protokol eğitim kümesini küçülttüğü için farkı yalnızca mesafe etkisine atfetmiyoruz.

{table(buffer[['year','target','model','log_r2_mean','raw_r2_mean','raw_rmse_mean','raw_mae_mean']])}

{table(buffer_effects)}

**Bağlam kazancı tamponlu testte korunuyor:** log hedefte 2018 merkez/çevre RF/CNN log R² yaklaşık **0,337/0,373/0,387**; 2022 için **0,314/0,352/0,363**. Bununla birlikte 2018'de log hedefli CNN'nin ham RMSE'si çevre RF'den biraz daha yüksek (47,77 /47,63 g/kg). 2022'de CNN ham RMSE'si çevre RF'den daha düşük (47,26 /48,11 g/kg) ve üç tekrarın hepsinde iyileşme var. Bu sonuçlar, **görüntü bağlamının katkısını** CNN'nin her koşulda en iyi model olduğu iddiasından daha güçlü destekliyor.

Yukarıdaki eşleşmiş farklar tamponlu protokolün içinde, aynı üç tekrar üzerinde hesaplanır. Ana protokol beş, tamponlu protokol üç tekrar içerir; bunların ortalamalarını karşılaştırırken tekrar sayısı farkı belirtilmelidir.

{table(targets.query("protocol=='block_100_buffer20' and model=='M6'")[['year','method','raw_r2_mean','raw_rmse_mean','raw_mae_mean','log_r2_mean']])}

Katlama başlangıcı/100 km grid sabittir. Katlama tekrarları örtüşen eğitim verileri kullanır; beş tekrar bağımsız deney değildir. Veri sayısı, yüksek SOC noktalarının azlığı, tek ülke ve tek hedef ölçüm yılı, belirlenmiş görüntü parçası büyüklüğü ve sabit model ayarları ana sınırlardır. Tek bir küçük CNN sonucundan bütün CNN mimarileri için genelleme yapılmaz. 20 m örnekleme SRTM/ERA5 ürünlerine yeni ince çözünürlüklü bilgi eklemez; iklim bu nedenle ayrı skalar girdiler olarak tutuldu.

## Makale için kullanılacak çerçeve

Yeni katkı iddiası, elde edilen sonuçlara bağlı **girdi bağlamı, hedef ölçeği ve doğrulama tasarımının birlikte incelenmesi** olmalı. Yalnızca PyTorch kullanılması veya RF'ye bir CNN eklenmesi yöntem yeniliği değildir. Hakemlerin örneklem, hedef derinliği, ham ölçek hataları ve doğrulama ayrımlarına ilişkin haklı eleştirileri bu deneylerde somut olarak ele alındı.

Makalenin ana tabloları aynı 663 nokta ve aynı fold'lar üzerinde düzenlenmeli; konsantrasyon/stok ve predictor yılı/SOC değişimi ayrımları korunmalı. CNN sonuçları, yalnızca olumluysa ana iddia haline getirilmemeli; çevre RF ve merkez MLP kontrolleriyle birlikte anlatılmalı. CNN'nin sınırlı veya negatif sonucu da bağlam bilgisinin hangi koşullarda işe yaradığı araştırma sorusunun parçasıdır. Dergi kapsamına göre bu bulguların bilgi işleme/yeniden üretilebilirlik veya pedometrik yorum ağırlığı değişebilir; ana araştırma sorusu değiştirilmemeli.

## PyTorch öğrenme dosyası

`cnn/SOC_PyTorch_walkthrough.ipynb`, gerçek veri şekillerini, merkez piksel/patch ayrımını, CNN ileri geçişini, tek bir eğitim batch'ini ve kaydedilmiş iç doğrulama eğrisini gösterir. Öğretici batch dış test skorlarını model seçimi için kullanmaz ve rapordaki bilimsel sonuçların yerine geçmez. Gerçek deney kodu `cnn/train_patch_models.py`; tekrarlı çalıştırma `cnn/run_batch.py`.

## Yeniden üretim

- RF hedef tanısı: `python target_compare.py --workers 6`.
- Patch hazırlığı: `python cnn/prepare_patches.py --csv2018 cnn/exports/SOC_CNN_patch17_2018.csv --csv2022 cnn/exports/SOC_CNN_patch17_2022.csv`.
- Aynı PyTorch ortamında bütün CNN kontrolleri: `python cnn/run_batch.py --workers 5`. Mevcut, aynı fingerprint'i taşıyan model–fold kayıtları yeniden kullanılır. Farklı ayarlarda ayrı çıktı dizini gerekir.
- Tamponlu CNN kontrolleri: `python cnn/run_batch.py --workers 5 --protocol block_100_buffer20 --repeats 3`.
- Rapor/şekil üretimi: `python build_stage3_report.py`.

Paket, girdileri, iki gerçek CSV/NPZ raster patch dosyasını, GEE kodunu, sabit dış fold'ları, iç katlama/katsayı kayıtlarını, bütün held-out tahminleri, ağ ağırlıkları ve normalizasyonlarını, öğretici notebook'u, sürüm bilgilerini ve şekilleri içerir. Çalışma ortamı ve yükleme yardımcıları pakete dahil değildir. Önceki aşamaların status dosyaları tarihsel checkpoint'tir; güncel durum `stage3_completion.json` dosyasıdır.

Yöntem kaynakları: Duan (1983), DOI [10.1080/01621459.1983.10478017](https://doi.org/10.1080/01621459.1983.10478017); [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html); [GEE sampleRegions](https://developers.google.com/earth-engine/apidocs/ee-image-sampleregions); [GEE neighborhoodToArray](https://developers.google.com/earth-engine/apidocs/ee-image-neighborhoodtoarray); [PyTorch kurulum](https://docs.pytorch.org/get-started/locally/).
'''
    (ART/'SOC_stage3_report_TR.md').write_text(report,encoding='utf-8')

def notebook():
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},id=f'soc-cell-{len(cells)}',source=s.splitlines(True)))
    def code(s):cells.append(dict(cell_type='code',metadata={},id=f'soc-cell-{len(cells)}',source=s.splitlines(True),execution_count=None,outputs=[]))
    md('# Gerçek SOC görüntü parçalarıyla PyTorch\n\nBu kısa dosya, tamamlanmış deneyin nasıl çalıştığını adım adım gösterir. Veriler gerçek GEE exportlarıdır. Raporun bilimsel skorları bütün dış test katlamalarından gelir; buradaki tek batch öğretici bir örnektir.')
    code("from pathlib import Path\nimport sys, json\nimport numpy as np, pandas as pd\nimport torch\nfrom torch import nn\nimport matplotlib.pyplot as plt\nCNN_DIR = Path.cwd()\nif not (CNN_DIR / 'data').exists():\n    CNN_DIR = CNN_DIR / 'cnn'\nPROJECT = CNN_DIR.parent\nsys.path.insert(0, str(CNN_DIR))\nfrom train_patch_models import PatchCNN, normalization, tensors, seed_all\nd = dict(np.load(CNN_DIR / 'data/patches_2018.npz', allow_pickle=False))\nprint('Torch:', torch.__version__)\nprint('Raster:', d['X'].shape, 'Climate:', d['climate'].shape, 'Target:', d['y'].shape)")
    md('## Tensor boyutları\n\n`[663, 14, 17, 17]` = 663 toprak noktası ×14 raster değişkeni ×17 satır ×17 sütun. CNN girişine geçerlilik maskesi eklenince kanal sayısı 15 olur. Her piksel 20 m; görüntü parçası 340 m genişliğindedir. Üç iklim değişkeni ayrı bir vektördedir; bu boyutta ERA5-Land texture bilgisi ürettiğimiz iddiası yoktur.')
    code("print(d['bands'])\nprint(d['climate_names'])\ni = 0\nrgb = np.moveaxis(d['X'][i, [2, 1, 0]], 0, -1)\nplt.imshow(np.clip(rgb / 0.30, 0, 1))\nplt.plot(8, 8, '+', color='white')\nplt.title(f\"Point {d['point_id'][i]} | SOC={d['y'][i]:.1f} g/kg\")\nplt.axis('off');")
    md('## Eğitim ve test önce ayrılır\n\nÖnceki RF ile aynı seed=0/fold=0 dış ayrımını okuyalım. Test noktaları normalizasyon, epoch seçimi veya bu öğretici eğitim batch’ine girmez.')
    code("f = pd.read_csv(PROJECT / 'stage2/results/fold_assignments.csv', dtype={'point_id': str})\nf = f.query(\"cohort == 'global_0_20' and protocol == 'block_100' and seed == 0 and fold == 0\")\nlookup = {p:i for i,p in enumerate(d['point_id'])}\ntrain = np.array([lookup[p] for p in f.loc[f.role.eq('train'), 'point_id']])\ntest = np.array([lookup[p] for p in f.loc[f.role.eq('test'), 'point_id']])\nassert not set(train) & set(test)\nprint(len(train), 'training points;', len(test), 'test points')")
    md('## Küçük CNN\n\nİki convolution katmanı komşu piksellerin yerel desenlerini işler. Global havuzlama bağlamı özetler. Merkez piksel değerleri ve iklim skaları bu özetle birleştirilir. Ağın çıktısı eğitim kümesinin ortalama/standart sapmasıyla ölçeklenmiş tek SOC hedefidir.')
    code("seed_all(42)\ntorch.set_num_threads(1)\nmodel = PatchCNN()\nprint(model)\nprint('Parameters:', sum(p.numel() for p in model.parameters()))\nnorm = normalization(d, train, target='log', kind='patchCNN')\nimage, point, target = tensors(d, train[:32], norm, target='log')\nprint('CNN image:', image.shape, 'center + climate:', point.shape)\nprediction = model(image, point)\nprint('Output:', prediction.shape)")
    md('## Tek bir eğitim adımı\n\n`zero_grad` eski gradyanları temizler; `backward` hatanın ağırlıklara göre türevini hesaplar; `step` ağırlıkları günceller. Bu örnekteki tek batch kaybı bir doğrulama sonucu değildir.')
    code("optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)\noptimizer.zero_grad()\nloss = nn.functional.mse_loss(prediction, target)\nloss.backward()\noptimizer.step()\nprint('Training batch loss:', float(loss.detach()))")
    md('## Gerçek deneyde epoch nasıl seçildi?\n\nİç mekânsal doğrulama eğrisini kaydedilmiş sonuçtan okuyalım. En iyi epoch yalnızca dış eğitim kümesinin içindeki doğrulama noktalarına göre seçildi. Daha sonra model tüm dış eğitim noktalarıyla yeniden kuruldu.')
    code("r = json.loads((CNN_DIR / 'results/2018/patchCNN_log_s0_f0.json').read_text())\ncurve = pd.DataFrame(r['inner_curve'])\nplt.plot(curve.epoch, curve.inner_validation_MSE_standardized)\nplt.axvline(r['selected_epochs'], color='orange', linestyle='--')\nplt.xlabel('Epoch'); plt.ylabel('Inner spatial validation MSE')\nprint('Selected epochs:', r['selected_epochs'])")
    md('## Sonuçlar\n\nHam SOC ve log ölçekteki başarıyı birlikte okuyalım. Tek bir fold veya olumlu sonuç seçmek yerine beş tekrarın pooled skorlarını kullanıyoruz. CNN’in karşısında hem merkez RF hem çevre özetlerini kullanan RF var.')
    code("s = pd.read_csv(CNN_DIR / 'performance_summary.csv')\ns[['year','target','model','raw_r2_mean','raw_rmse_mean','raw_mae_mean','log_r2_mean']]")
    md('## Bir sonraki deneyi değiştirirken\n\nYeni band, patch büyüklüğü veya ağ ayarı denemesi için yeni bir çıktı dizini kullan. Dış test noktalarıyla normalizasyon veya erken durdurma yapma. SOC ≥150 g/kg bir organik toprak tanımı değildir. Predictor yılı değişince hedef ölçüm yılı değişmez. Öğretici notebook’u çalıştırmak rapor skorlarını güncellemez; bunun için ana runner ve tam katlamalar gerekir.')
    nb=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3 (PyTorch)',language='python',name='python3'),language_info=dict(name='python',version='3.12')),nbformat=4,nbformat_minor=5)
    (ROOT/'cnn/SOC_PyTorch_walkthrough.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2))

def create_package():
    package=ART/'SOC_stage3_package.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(ROOT.rglob('*')):
            if not p.is_file() or p==package or any(x in p.parts for x in ['helpers','torch_env','__pycache__']):continue
            if p.suffix=='.tmp':continue
            z.write(p,p.relative_to(ROOT))
    with zipfile.ZipFile(package) as z:assert z.testzip() is None
    return package

def main():
    status=json.loads((ROOT/'results/completion.json').read_text());assert status['all_completed'] and status['RF_fits']==1550
    targets=pd.read_csv(ROOT/'results/performance_summary.csv')
    pooled,pred,cnn,training,effects,parity,validation=cnn_summary()
    buffer,buffer_effects=buffered_summary()
    figures(targets,cnn,buffer);notebook();report(targets,pooled,cnn,training,effects,parity,validation,buffer,buffer_effects)
    final=dict(target_RF_fits=1550,CNN_control_model_fold_fits=640,CNN_trained=True,years=[2018,2022],n_0_20cm_points=663,
        target_year=2018,primary_protocol='100km spatial blocks, 5 repeats x 5 folds',
        patch_exports={2018:dict(task_id='VZN3TIXTQOOZ7OQWC7OS2DS2',phase='Completed',runtime_seconds=39),2022:dict(task_id='CGNCY4EF2N7PQII5SVFA5TDY',phase='Completed',runtime_seconds=46)},
        patch_qc=[{k:v for k,v in d.items() if k!='cnn_trained'} for d in validation],center_RF_reference_parity_max_abs_error=parity,cnn_buffer20_run=True,
        target_baseline_parity=json.loads((ROOT/'results/completion.json').read_text())['baseline_reproduction_max_abs_error'])
    (ROOT/'stage3_completion.json').write_text(json.dumps(final,indent=2))
    versions=json.loads((ROOT/'results/design.json').read_text())['software']
    (ROOT/'requirements_rf.txt').write_text('\n'.join(k+'=='+versions[k] for k in ['numpy','pandas','scipy'])+'\nscikit-learn=='+versions['sklearn']+'\nmatplotlib=='+matplotlib.__version__+'\n')
    (ROOT/'requirements_cnn.txt').write_text('-r requirements_rf.txt\n--extra-index-url https://download.pytorch.org/whl/cpu\ntorch==2.14.1+cpu\n')
    (ROOT/'README.md').write_text('''# SOC stage 3: real-patch CNN and target-scale diagnostics\n\nRead deliverables/SOC_stage3_report_TR.md first. This package contains completed scientific experiments; historical stage-1/stage-2 status files refer to their original checkpoints. Current status: stage3_completion.json. Both predictor years use 2018 SOC labels, depth 0–20 cm, n=663.\n\nRun RF diagnostics with `python target_compare.py --workers 6`; run validated patch-model experiments with `python cnn/run_batch.py --workers 5`; run buffered patch-model checks with `python cnn/run_batch.py --workers 5 --protocol block_100_buffer20 --repeats 3`; rebuild report with `python build_stage3_report.py`. The included NPZ arrays are sufficient for training; the real CSV exports are retained for traceability. No Earth Engine sign-in is needed for local reproduction. GEE code is included if source extraction must be repeated.\n\nThe Linux CPU requirements file records the exact tested PyTorch version. Other platforms should use PyTorch's official install selector and record their versions. The workspace environment and package-upload helpers are excluded from this archive. The teaching notebook is cnn/SOC_PyTorch_walkthrough.ipynb.\n\nFive repetitions share training points; their SD is not a confidence interval. Outer folds were reused after inspecting previous errors: this is a diagnostic follow-up, not independent external confirmation. Patch-model tests use both 100km spatial folds (5 repetitions) and a 20km buffered version (3 repetitions).\n''')
    package=create_package()
    print('Center RF parity:',parity)
    print(cnn[['year','target','model','raw_r2_mean','raw_rmse_mean','raw_mae_mean','log_r2_mean']].round(4).to_string(index=False))
    print('Paired changes:\n',effects.round(4).to_string(index=False))
    print('Package:',package.stat().st_size,'bytes')

if __name__=='__main__':main()
