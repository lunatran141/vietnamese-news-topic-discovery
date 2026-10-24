"""4 NMF figures from frozen v1 artifacts; no model fitting."""
import argparse, hashlib, json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[2]
RUN='nmf_p3_k10_seed42_v1'
BLUE,TEAL,ORANGE,INK,MUTED,LIGHT='#245C9C','#087F8C','#D78125','#182B42','#52657B','#EFF4F8'
SOURCES={'config':'configs/nmf.yaml','selection':'results/nmf/nmf_p3_select_review_seed42_v1/topic_count_metrics.csv','metrics':f'results/nmf/{RUN}/test_metrics.json','keywords':f'artifacts/nmf/{RUN}/topic_keywords.json','cases':f'results/nmf/{RUN}/multi_topic_cases.csv','assignments':f'artifacts/nmf/{RUN}/document_topics.parquet'}
def canvas(title,subtitle):
 f=plt.figure(figsize=(16,9),facecolor='white');f.text(.055,.935,title,fontsize=27,weight='bold',color=INK);f.text(.055,.887,subtitle,fontsize=14,color=MUTED);return f
def foot(f,text):f.text(.055,.035,text,fontsize=10,color=MUTED,va='bottom')
def blank(f):
 a=f.add_axes([0,0,1,1]);a.set(xlim=(0,1),ylim=(0,1));a.axis('off');return a
def box(a,x,y,w,h,color=LIGHT,edge='none'):a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.012',facecolor=color,edgecolor=edge))
def method():
 f=canvas('NMF: từ TF-IDF đến các chủ đề','Sơ đồ nguyên lý X ≈ W × H — W và H có các phần tử không âm');a=blank(f)
 for x,y,w,h,n,label,size,rows,cols,col in [(.07,.43,.25,.29,'X','Bài báo × Đặc trưng','D × V',6,10,BLUE),(.43,.43,.16,.29,'W','Bài báo × Chủ đề','D × K',6,4,TEAL),(.70,.43,.24,.29,'H','Chủ đề × Đặc trưng','K × V',4,10,ORANGE)]:
  box(a,x,y,w,h,edge=col)
  for i in range(1,rows):a.plot([x,x+w],[y+h*i/rows]*2,color='white',lw=2)
  for j in range(1,cols):a.plot([x+w*j/cols]*2,[y,y+h],color='white',lw=2)
  a.text(x+w/2,y+h/2,n,ha='center',va='center',fontsize=43,weight='bold',color=col);a.text(x+w/2,y-.055,label,ha='center',fontsize=16);a.text(x+w/2,y-.095,size,ha='center',fontsize=15,color=MUTED)
 a.text(.373,.575,'≈',fontsize=42,ha='center',va='center');a.text(.646,.575,'×',fontsize=37,ha='center',va='center')
 for x,t,c in [(.07,'TF-IDF đầu vào',BLUE),(.43,'Trọng số chủ đề',TEAL),(.70,'Từ khóa chủ đề',ORANGE)]:a.text(x,.78,t,fontsize=17,weight='bold',color=c)
 box(a,.065,.13,.87,.12);a.text(.085,.209,'P3: chuẩn hóa · tách từ PyVi · loại stopword',fontsize=15);a.text(.085,.166,'TF-IDF unigram + bigram · tối đa 50.000 đặc trưng',fontsize=15)
 a.text(.935,.098,'Coordinate Descent · Frobenius loss · tối đa 500 vòng · seed 42',ha='right',fontsize=12,color=MUTED)
 foot(f,'Sơ đồ minh họa: số ô không biểu diễn dữ liệu thực. D: số bài; V: số đặc trưng; K: số chủ đề. Cấu hình triển khai: configs/nmf.yaml.');return f

def select_k(t):
 f=canvas('Chọn K=10 theo kết quả Validation','Học trên Train; đánh giá trên Validation — Test không tham gia lựa chọn K')
 for i,(key,title,col,lim) in enumerate([('npmi_coherence','NPMI coherence ↑',BLUE,(.33,.47)),('topic_diversity','Topic diversity ↑',TEAL,(.97,1.006)),('normalized_reconstruction_error','Sai số tái tạo chuẩn hóa ↓',ORANGE,(.953,.976))]):
  a=f.add_axes([.065+i*.317,.30,.26,.48]);a.axvspan(9.65,10.35,color=TEAL,alpha=.10);a.plot(t.n_topics,t[key],marker='o',color=col,lw=2.4,ms=7)
  for k,v in zip(t.n_topics,t[key]):a.annotate(f'{v:.4f}',(k,v),xytext=(0,11),textcoords='offset points',ha='center',fontsize=11)
  a.set(title=title,xlabel='Số chủ đề K',xticks=t.n_topics,ylim=lim,xlim=(3.5,12.5));a.tick_params(labelsize=11);a.grid(axis='y',alpha=.2);a.spines[['top','right']].set_visible(False);a.ticklabel_format(axis='y',style='plain',useOffset=False)
 f.text(.065,.17,'K=10: NPMI cao nhất (0,4493)\nTopic diversity = 1,0000',fontsize=17,weight='bold',color=BLUE);f.text(.57,.17,'K=12: sai số tái tạo thấp nhất (0,9568)\nLựa chọn phụ thuộc tiêu chí ưu tiên',fontsize=16)
 foot(f,'Ưu tiên chọn cấu hình: NPMI → topic diversity → sai số tái tạo. Chưa phải tối ưu theo review độc lập.\nNguồn: results/nmf/nmf_p3_select_review_seed42_v1/topic_count_metrics.csv.');return f

def final_results(m,kw):
 f=canvas('NMF Final: kết quả Test và từ khóa thực tế','K=10 · seed 42 · học lại trên 7.884 bài Train+Validation · Test: 1.390 bài');a=blank(f)
 for i,(label,key) in enumerate([('NPMI','npmi_coherence'),('Topic diversity','topic_diversity'),('ARI','adjusted_rand_index'),('NMI','normalized_mutual_info')]):
  x,y=.06+(i%2)*.20,.56-(i//2)*.24;box(a,x,y,.18,.18);a.text(x+.018,y+.128,label,fontsize=15,color=MUTED);a.text(x+.018,y+.048,f'{m[key]:.6f}',fontsize=25,weight='bold',color=BLUE)
 for i,(tid,name,col) in enumerate([('NMF_02','Điều tra / pháp luật',BLUE),('NMF_03','Y tế',TEAL),('NMF_04','Giáo dục',ORANGE)]):
  words=kw[tid]['keywords'][:5];b=f.add_axes([.64,.625-i*.225,.30,.145]);values=[w['weight'] for w in words];b.barh([w['term'].replace('_',' ') for w in words],values,height=.66,color=col);b.invert_yaxis();b.set_xlim(0,.76)
  for j,v in enumerate(values):b.text(v+.012,j,f'{v:.3f}',va='center',fontsize=10)
  b.set_title(f'{tid} — {name} (đề xuất)',fontsize=13,loc='left');b.set_xticks([0,.2,.4,.6]);b.tick_params(labelsize=10);b.spines[['top','right']].set_visible(False)
 a.text(.06,.217,'ARI / NMI: đối chiếu với chuyên mục báo;\nkhông phải độ chính xác phân loại.',fontsize=14);a.text(.06,.145,'Diversity = 1: top-10 đặc trưng không trùng;\nkhông chứng minh khác biệt hoàn toàn về ngữ nghĩa.',fontsize=13,color=MUTED)
 foot(f,'Thanh ngang: trọng số H thực tế, dùng xếp hạng từ trong từng topic; nhãn chưa được review độc lập.\nNguồn: test_metrics.json và topic_keywords.json, run nmf_p3_k10_seed42_v1.');return f

def multi_cases(cases,n,total,threshold):
 f=canvas('Bài đa chủ đề: minh họa và trường hợp cần kiểm tra',f'{n}/{total} bài Test là ứng viên — topic share lớn thứ hai ≥ {threshold:.2f}; chưa xác nhận đa chủ đề');a=blank(f)
 examples=[('VNN_93c346ff1800','Khám xét nhà nữ diễn viên 37 tuổi','NMF_05 — Giải trí','Giao thoa có căn cứ sơ bộ','Cần xác nhận cả hai mảng có phải\ntrọng tâm nội dung hay không.'),('VNE_1e32ccbe78ad','Green SM mở cơ hội gia nhập\ncho tài xế mới','NMF_00 — Doanh nghiệp','Topic pháp luật có dấu hiệu lệch nội dung','Hai trọng số gần nhau chưa đủ\nđể xác nhận bài có nhiều chủ đề.')]
 for i,(aid,title,secondary,verdict,note) in enumerate(examples):
  x=.06+i*.47;box(a,x,.19,.42,.61);r=cases.loc[cases.article_id.eq(aid)].iloc[0];p,q=float(r.primary_topic_share),float(r.secondary_topic_share);values=[p,q,1-p-q];a.text(x+.02,.75,title,fontsize=17,weight='bold',va='top');a.text(x+.02,.65,aid,fontsize=11,color=MUTED)
  b=f.add_axes([x+.02,.51,.38,.075]);left=0
  for v,col in zip(values,[BLUE,ORANGE,'#D9E3EC']):b.barh([0],[v],left=left,color=col,height=.7);b.text(left+v/2,0,f'{v*100:.2f}%',ha='center',va='center',fontsize=12,color='white' if col!='#D9E3EC' else INK,weight='bold');left+=v
  b.set(xlim=(0,1),xticks=[0,.5,1],xticklabels=['0%','50%','100%'],yticks=[]);b.spines[['top','left','right']].set_visible(False)
  for y,text,col,size in [(.435,f'NMF_02 — Điều tra / pháp luật: {p:.4f}',BLUE,13),(.39,f'{secondary}: {q:.4f}',ORANGE,13),(.345,f'Xám: các topic còn lại ({values[2]:.4f})',MUTED,12)]:a.text(x+.02,y,text,fontsize=size,color=col)
  a.text(x+.02,.29,verdict,fontsize=12,weight='bold');a.text(x+.02,.265,note,fontsize=12,color=MUTED,va='top')
 foot(f,'Topic share là trọng số tương đối, không phải xác suất nhãn đúng. Nhận xét sơ bộ còn chờ reviewer độc lập.\nNguồn: multi_topic_cases.csv và document_topics.parquet, run nmf_p3_k10_seed42_v1.');return f

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('results/nmf/slide_figures'));args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'Arial','font.size':12,'axes.titlesize':15,'axes.labelcolor':INK,'text.color':INK,'svg.fonttype':'none'})
 cfg=yaml.safe_load((ROOT/SOURCES['config']).read_text(encoding='utf-8-sig'));t=pd.read_csv(ROOT/SOURCES['selection']).sort_values('n_topics');m=json.loads((ROOT/SOURCES['metrics']).read_text(encoding='utf-8-sig'));kw={r['topic_id']:r for r in json.loads((ROOT/SOURCES['keywords']).read_text(encoding='utf-8-sig'))};cases=pd.read_csv(ROOT/SOURCES['cases']);docs=pd.read_parquet(ROOT/SOURCES['assignments']);n=int(docs.is_multi_topic.sum())
 assert t.n_topics.tolist()==[4,6,8,10,12] and int(t.loc[t.npmi_coherence.idxmax(),'n_topics'])==10
 assert m['n_topics']==10 and len(docs)==1390 and n==523 and docs.article_id.is_unique
 ids=['VNN_93c346ff1800','VNE_1e32ccbe78ad']
 for aid in ids:
  row=cases.loc[cases.article_id.eq(aid)];assert len(row)==1;values=row[[f'topic_{i:02d}_share' for i in range(10)]].to_numpy(dtype=float);assert np.all(values>=0) and np.allclose(values.sum(axis=1),1)
 figs={'01_nmf_method':method(),'02_nmf_select_k':select_k(t),'03_nmf_test_keywords':final_results(m,kw),'04_nmf_multi_topic_cases':multi_cases(cases,n,len(docs),cfg['analysis']['secondary_topic_min_share'])};files=[]
 for name,fig in figs.items():
  for ext in ['png','svg']:p=out/f'{name}.{ext}';fig.savefig(p,dpi=180,facecolor='white');files.append(p)
  plt.close(fig)
 data={'run_id':RUN,'figure_01':'Conceptual schematic; matrix cells are not measured data.','selection':t.to_dict(orient='records'),'test_metrics':m,'keywords':[kw[k] for k in ['NMF_02','NMF_03','NMF_04']],'cases':cases.loc[cases.article_id.isin(ids),['article_id','title','primary_topic_share','secondary_topic_share']].to_dict(orient='records'),'test_document_count':len(docs),'candidate_count':n,'sources':{k:{'path':v,'sha256':hashlib.sha256((ROOT/v).read_bytes()).hexdigest()} for k,v in SOURCES.items()}}
 p=out/'figure_data.json';p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');files.append(p)
 text='''# NMF — bộ hình cho bốn slide
01_nmf_method: sơ đồ nguyên lý; số ô chỉ minh họa.
02_nmf_select_k: đủ năm giá trị K trên Validation, đọc từ CSV thực nghiệm.
03_nmf_test_keywords: metric Final Test và top-5 từ khóa H thực tế; nhãn đề xuất.
04_nmf_multi_topic_cases: hai case Test; có phần các topic còn lại, tổng share bằng 1.
PNG: 2880 × 1620 (16:9). SVG: vector, font Arial.
Số liệu và SHA256 nguồn: figure_data.json. Không huấn luyện lại mô hình.
523 bài là ứng viên đa chủ đề chưa xác nhận. Nhận xét sơ bộ cần review độc lập.
Các hình dùng run nmf_p3_k10_seed42_v1; không trộn với dashboard_v2.
Tạo lại từ thư mục gốc repo: python -B -m experiments.nmf.plot_presentation_figures
'''
 p=out/'README.md';p.write_text(text,encoding='utf-8');files.append(p)
 with ZipFile(out/'nmf_slide_images.zip','w',compression=ZIP_DEFLATED) as z:
  for p in files:z.write(p,p.name)
 print(json.dumps({'output':str(out),'figures':len(figs),'files':[p.name for p in files]},ensure_ascii=True))
if __name__=='__main__':main()
