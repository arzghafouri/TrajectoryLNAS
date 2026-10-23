# TrajectoryLNAS: Trajectory Prediction Architecture Optimization via Lightweight Neural Architecture Search
**پروژه پایان‌نامه کارشناسی ارشد مهندسی کامپیوتر - هوش مصنوعی و رباتیک**  
دانشجو: علیرضا غفوری (شماره دانشجویی: 4031631209)  
استاد راهنما: دکتر زینب قاسمی نراقی | استاد مشاور: دکتر زینب عابدینی  
شماره ثبت پژوهش: ۲۸۲۴۵۲۵۹ | شماره ثبت ایرانداک: ۱۹۸۷۵۶۸

---

## ساختار پوشه‌ها و فایل‌ها (Codebase Structure)
```
python_src/
├── data/
│   ├── argoverse_dataset.py   # لودر اختصاصی Argoverse v1 (خواندن CSVها، دوران خودرومحور، پلی‌لاین‌های معابر)
│   ├── nuscenes_dataset.py    # لودر اختصاصی nuScenes Prediction Benchmark (درون‌یابی به 10Hz و تبدیل مختصات)
│   └── data_factory.py        # کارخانه یکپارچه DataLoader و تنظیم آدرس‌های پیش‌فرض
├── models/
│   ├── supernet_operators.py  # 11 عملگر سه‌گانه زمانی، تعاملی و نقشه + MixedOps
│   ├── search_supernet.py     # شبکه فراگیر (Super-Net) و پارامترهای معماری ArchParams
│   └── final_model.py         # معماری استخراج‌شده گسسته (FinalTrajectoryModel)
├── utils.py                   # معیارهای ADE, FDE, Miss Rate, پروکسی‌های بدون هزینه (SNIP, SynFlow)
├── search.py                  # فرآیند بهینه‌سازی دوسطحی مشتق‌پذیر DARTS با جریمه FLOPs
├── train_final.py             # آموزش مدل نهایی استخراج‌شده از صفر با یادگیری بازپخش و کسینوسی
├── evaluate.py                # مقایسه عملکرد با TrajectoryNAS, LaneGCN, Trajectron++
└── requirements.txt           # وابستگی‌های مورد نیاز
```

---

## آدرس‌ها و راهنمای دیتاست‌ها (Datasets & Directory Paths)

### ۱. دیتاست Argoverse 1 Motion Forecasting
* **لینک دریافت رسمی**: [https://www.argoverse.org/av1.html](https://www.argoverse.org/av1.html)
* **ساختار پوشه‌ها (Folder Hierarchy)**:
  ```
  data/argoverse/
  ├── train/data/
  │   ├── 1.csv
  │   ├── 2.csv
  │   └── ... (205,942 توالی)
  ├── val/data/
  │   ├── 100.csv
  │   └── ... (39,472 توالی)
  └── map_files/
      ├── pit_centerline.npy
      └── mia_centerline.npy
  ```
* **ابعاد داده**: ۵۰ گام زمانی (۲۰ گام گذشته @ 10Hz = ۲٫۰ ثانیه، ۳۰ گام آینده @ 10Hz = ۳٫۰ ثانیه).
* **تبدیل مختصات**: انتقال مبدأ به نقطه $t=19$ خودروی هدف و چرخش بردار سرعت به سمت محور مثبت X.

### ۲. دیتاست nuScenes Motion Prediction
* **لینک دریافت رسمی**: [https://www.nuscenes.org/download](https://www.nuscenes.org/download)
* **ساختار پوشه‌ها (Folder Hierarchy)**:
  ```
  data/nuscenes/
  ├── v1.0-trainval/
  │   ├── sample.json
  │   ├── sample_data.json
  │   ├── instance.json
  │   └── ...
  ├── maps/
  │   ├── boston-seaport.json
  │   └── singapore-onenorth.json
  └── prediction/
      ├── train_instances.json
      └── val_instances.json
  ```
* **ابعاد داده**: فریم‌های ۲ هرتز nuScenes به صورت خودکار به فرکانس استاندارد ۱۰ هرتز (۲۰ گذشته، ۳۰ آینده) نگاشت می‌شوند.

---

## نحوه اجرا (How to Run)

### ۱. نصب وابستگی‌ها:
```bash
pip install -r requirements.txt
```

### ۲. اجرای خط‌لوله کامل ۳ مرحله‌ای (پیشنهادی):
اسکریپت ارکستراسیون یکپارچه `main.py` به صورت خودکار ۳ فاز زیر را متوالی اجرا می‌کند:
* **مرحله ۱:** جستجوی معماری پیوسته DARTS (پیش‌فرض ۳۰ اپوک).
* **مرحله ۲:** آموزش مستقل مدل استخراج‌شده از صفر به همراه پایش توقف زودهنگام (Early Stopping با Patience=8).
* **مرحله ۳:** ارزیابی آکادمیک روی ۶۷ هزار نمونه، استخراج سنجه‌های سخت‌افزاری (FPS و تاخیر)، مقایسه SOTA و رسم نمودارهای گزارش.

```bash
# اجرا روی nuScenes با پارامترهای بهینه‌شده برای کارت گرافیک
python main.py --dataset nuscenes --search_epochs 30 --final_epochs 50 --batch_size 32

# اجرا روی Argoverse
python main.py --dataset argoverse --search_epochs 30 --final_epochs 50 --batch_size 32
```

### ۳. اجرای مستقل هر مرحله به صورت ماژولار:

* **فقط جستجوی معماری سوپرنت (Stage 1):**
```bash
python train/search.py --dataset nuscenes --epochs 30 --batch_size 32
```

* **فقط آموزش مدل نهایی استخراج‌شده با Early Stopping (Stage 2):**
```bash
python train/train_final.py --dataset nuscenes --epochs 50 --patience 8 --batch_size 32
```

* **فقط ارزیابی و مقایسه بنچمارک روی چک‌پوینت ذخیره‌شده (Stage 3):**
```bash
python evaluate/evaluator.py --dataset nuscenes --checkpoint checkpoints/best_trajectory_lnas.pt --batch_size 32
```

---

## خروجی‌ها و گزارش‌های تولیدشده (Generated Artifacts)
پس از اجرای موفق، کلیه نتایج به صورت خودکار در مسیرهای زیر ذخیره می‌شوند:
1. `checkpoints/best_trajectory_lnas.pt`: اوزان بهینه استخراج‌شده از بهترین اپوک مدل نهایی.
2. `report/search_execution.log`: لاگ متنی کامل اجرای خط‌لوله به همراه آلفاهای هر اپوک.
3. `report/real_experiment_results.json`: فایل ساختاریافته شامل معیارهای ZCP، آلفاهای ۳۰ اپوک، تاریخچه آموزش و پروفایل سخت‌افزاری (جهت آپلود مستقیم در وب‌اپلیکیشن).
4. `report/plots/horizon_error_curve.png`: نمودار خطای جابه‌جایی در افق‌های ۱، ۲ و ۳ ثانیه.
5. `report/plots/sota_benchmark_comparison.png`: نمودار میله‌ای مقایسه عملکرد با مدلهای مطرح دنیا (Trajectron++, LaneGCN, TrajectoryNAS).
