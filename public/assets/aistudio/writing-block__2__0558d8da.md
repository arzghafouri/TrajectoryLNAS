# مأموریت: بازطراحی و توسعه prepare_data.py برای استخراج دیتاست استاندارد و پژوهش‌محور nuScenes در پروژه TrajectoryLNAS

به‌عنوان یک Senior Machine Learning Engineer و متخصص Autonomous Driving، PyTorch، nuScenes Prediction، Trajectory Forecasting، HD Map Processing و Neural Architecture Search، فایل فعلی `prepare_data.py` را بررسی و اصلاح کن.

مخزن پروژه:

https://github.com/arzghafouri/TrajectoryLNAS/tree/main/python_src

فایل هدف:

`python_src/data/prepare_data.py`

هدف اصلی، تبدیل فایل فعلی به یک ابزار Production-Ready برای استخراج، اعتبارسنجی، پردازش و ذخیره‌سازی داده‌های واقعی nuScenes است؛ به‌گونه‌ای که خروجی آن مستقیماً برای جستجوی معماری DARTS و آموزش مدل‌های پیش‌بینی مسیر قابل استفاده باشد.

---

## ۱. ابتدا ساختار واقعی پروژه را بررسی کن

قبل از هرگونه تغییر، فایل‌ها و وابستگی‌های زیر را بررسی کن:

- `prepare_data.py`
- `data_summery.py`
- Dataset classes
- DataLoader implementation
- `config/search_config.py`
- مدل نهایی Trajectory Prediction
- DARTS Searcher
- معماری‌های Temporal، Interaction و Map
- کد انتخاب Search Samples
- کد بارگذاری Train و Validation
- ساختار واقعی فایل‌های JSON خروجی

به‌خصوص مشخص کن مدل فعلی دقیقاً چه کلیدها، ابعاد Tensorها، Maskها و فرمت‌هایی را از Dataset انتظار دارد.

**اصل مهم:** خروجی جدید `prepare_data.py` باید با Dataset Loader و مدل موجود سازگار باشد. تغییر فرمت داده بدون بررسی تمام مصرف‌کنندگان آن ممنوع است.

از بازنویسی غیرضروری سایر بخش‌های پروژه خودداری کن.

---

## ۲. اصلاح استخراج داده‌های واقعی nuScenes

کلاس `NuScenesRealDataExtractor` را بازبینی و اصلاح کن.

### ۲.۱. استخراج صحیح Trajectory

برای هر عامل هدف، داده‌های زیر را استخراج کن:

- Past trajectory: 20 timesteps
- Future trajectory: 30 timesteps
- Sampling frequency: 10 Hz
- Target category
- Instance token
- Sample token
- Scene token
- Scene name
- Timestamp
- City / Location
- Agent position
- Agent heading
- Agent velocity، در صورت امکان محاسبه معتبر

قواعد مهم:

1. داده‌های Past و Future باید به ترتیب زمانی صحیح باشند.
2. هیچ timestamp تکراری یا نامعتبر نباید وارد مسیر شود.
3. موقعیت‌ها باید در یک دستگاه مختصات یکسان باشند.
4. زمان‌بندی باید واقعاً با نرخ 10Hz سازگار باشد.
5. در صورت کمبود داده، به‌جای تولید مسیر مصنوعی، نمونه را با علت مشخص رد کن.
6. Interpolation فقط زمانی انجام شود که از نظر زمانی و مکانی معتبر باشد.
7. مسیرهای ناقص را بدون ثبت علت حذف نکن.
8. هیچ مسیر آینده نباید از اطلاعاتی استفاده کند که بعد از زمان هدف در بخش Past قرار گرفته‌اند.

از interpolation ساده مبتنی بر index بدون بررسی timestamp اجتناب کن.

در صورت استفاده از interpolation، زمان واقعی هر مشاهده را مبنا قرار بده.

### ۲.۲. تبدیل دستگاه مختصات

برای تمام ورودی‌ها مشخص کن که مختصات در چه دستگاهی قرار دارند:

- Global coordinates
- Agent-centric coordinates
- Local map coordinates

در صورت نیاز، تبدیل مختصات را با استفاده از rotation و translation معتبر nuScenes انجام بده.

مختصات Trajectory، Neighbors و HD Map باید قبل از ورود به مدل با یکدیگر سازگار باشند.

تبدیل مختصات نباید باعث جابه‌جایی اشتباه Past، Future و Map شود.

در صورت استفاده از مختصات Agent-centric، مبدأ را موقعیت عامل هدف در زمان مشاهده تعریف کن و جهت محور‌ها را به‌صورت ثابت و مستند مشخص کن.

---

## ۳. اصلاح اساسی HD Map Extraction

این بخش بالاترین اولویت توسعه است.

گزارش آماری فعلی نشان داده است:

- HD Map Samples: صفر
- HD Map Lanes: صفر
- Map Points: صفر

درحالی‌که دیتاست اصلی nuScenes دارای اطلاعات نقشه HD است.

### ۳.۱. اصلاح Map Availability Validation

کلاس `NuScenesMapExpansionValidator` را بازطراحی کن.

وجود چند فایل JSON به‌تنهایی برای تأیید سلامت نقشه کافی نیست.

بررسی کن:

- فایل‌های نقشه مربوط به چهار شهر nuScenes وجود داشته باشند.
- فایل‌ها قابل خواندن باشند.
- ساختار لایه‌های نقشه معتبر باشد.
- NuScenesMap بتواند آن‌ها را بارگذاری کند.
- اطلاعات map records قابل بازیابی باشند.

شهرهای مورد بررسی:

- boston-seaport
- singapore-onenorth
- singapore-queenstown
- singapore-hollandvillage

در صورت خرابی یا فقدان نقشه یک شهر، فقط نمونه‌های همان شهر را تحت تأثیر قرار بده و علت را ثبت کن.

### ۳.۲. استفاده صحیح از nuScenes Map API

تابع فعلی:

`extract_map_lanes_real`

را به‌طور جدی بررسی کن.

در نسخه فعلی فقط از لایه‌های:

- lane_divider
- road_divider

استفاده شده است.

این لایه‌ها به‌تنهایی نماینده کامل ساختار خطوط حرکت خودرو نیستند.

در صورت نیاز و پشتیبانی API، لایه‌های زیر را نیز بررسی کن:

- lane
- lane_connector
- road_segment
- road_block
- drivable_area
- walkway
- pedestrian_crossing
- stop_line
- traffic_light

برای پیش‌بینی مسیر خودرو، اطلاعات lane و lane_connector اهمیت ویژه دارند.

برای پیش‌بینی عابر پیاده، walkway و pedestrian_crossing می‌توانند مفید باشند.

### ۳.۳. استخراج Polyline واقعی

هر Lane یا Map Element را به‌صورت Polyline معتبر استخراج کن.

الزامات:

- استخراج تمام نقاط هندسی موجود، نه صرفاً دو نقطه ابتدا و انتها.
- حفظ ترتیب مکانی نقاط.
- حذف نقاط نامعتبر.
- محاسبه طول واقعی Polyline.
- محاسبه فاصله Map Element تا عامل هدف.
- تعیین نوع Lane یا Map Element.
- حفظ شناسه عنصر نقشه، در صورت امکان.
- جلوگیری از تکرار Polylineها.
- محدود کردن تعداد عناصر نقشه با سیاستی قابل تنظیم.
- انتخاب عناصر بر اساس فاصله و ارتباط مکانی، نه ترتیب تصادفی.

اگر Map API برای یک لایه هندسه کامل ارائه می‌دهد، از همان API استفاده کن و آن را به دو نقطه ابتدا و انتها تقلیل نده.

### ۳.۴. ارتباط HD Map با Agent

Map باید در محدوده مکانی مشخصی از Agent هدف استخراج شود.

پارامترهای زیر قابل تنظیم باشند:

- Map search radius
- Maximum number of lanes
- Maximum points per lane
- Minimum lane length
- Maximum lane distance
- Map feature types

از داده‌های نقشه‌ای که با مختصات Trajectory سازگار نیستند استفاده نکن.

### ۳.۵. اعتبارسنجی Map

برای هر نمونه ثبت کن:

- تعداد Laneها
- تعداد Map Elements
- تعداد نقاط
- طول متوسط Lane
- فاصله نزدیک‌ترین Lane تا Agent
- نوع عناصر نقشه
- وضعیت اعتبار Map
- علت فقدان Map

در پایان، اگر HD Map برای هیچ نمونه‌ای استخراج نشد، ابزار باید با Warning جدی خاتمه یابد و اجازه ندهد Dataset به‌اشتباه به‌عنوان دیتاست کامل Map-aware معرفی شود.

**ممنوع:** ساخت Map مصنوعی برای جایگزین کردن Map واقعی در حالت real.

---

## ۴. اصلاح استخراج Neighbors و اطلاعات تعامل

گزارش فعلی نشان داده است:

- میانگین همسایگان: حدود 5
- میانگین فاصله همسایگان: 41.52 متر

این فاصله برای همسایگان مؤثر در تعامل حرکتی نیازمند بررسی است.

### ۴.۱. استخراج Neighbors

کد فعلی فقط Annotationهای موجود در همان Keyframe را بررسی می‌کند و پنج عامل اول را انتخاب می‌کند.

این روش را اصلاح کن.

برای هر عامل هدف:

1. تمام عوامل معتبر پیرامون را شناسایی کن.
2. فاصله آن‌ها را از Agent هدف محاسبه کن.
3. بر اساس فاصله، نزدیک‌ترین عوامل را انتخاب کن.
4. عامل هدف را از فهرست همسایگان حذف کن.
5. عوامل فاقد Instance یا Trajectory معتبر را حذف کن.
6. عوامل ثابت و غیرمتحرک را بر اساس سیاست قابل تنظیم مدیریت کن.
7. تعداد همسایگان را قابل تنظیم کن.
8. از انتخاب تصادفی همسایگان خودداری کن.

پارامترهای پیشنهادی:

- `max_neighbors`
- `neighbor_radius`
- `min_neighbor_history`
- `neighbor_selection_strategy`
- `include_static_agents`

### ۴.۲. استخراج زمانی همسایگان

همسایگان باید دارای Past trajectory معتبر باشند.

برای هر Neighbor:

- Past positions
- Relative positions
- Relative velocity
- Distance to target
- Category
- Validity mask

را استخراج کن.

تمام همسایگان باید در یک دستگاه مختصات مشترک با Agent هدف قرار داشته باشند.

اگر یک Neighbor در برخی timestepها مشاهده نشده است، از Mask معتبر استفاده کن؛ صفر را به‌عنوان داده واقعی یا داده معتبر فرض نکن.

### ۴.۳. Neighbor Mask

برای داده‌های همسایگان Mask مستقل ذخیره کن.

ساختار پیشنهادی:

`neighbors_past: [N_neighbors, 20, 2]`

`neighbors_mask: [N_neighbors, 20]`

`neighbor_categories: [N_neighbors]`

`neighbor_distances: [N_neighbors]`

ابعاد دقیق را با Dataset Loader موجود تطبیق بده.

Padding فقط برای یکسان‌سازی ابعاد انجام شود و با Mask از داده واقعی تفکیک گردد.

---

## ۵. اصلاح Data Splitting و جلوگیری از Data Leakage

این بخش برای اعتبار پژوهشی بسیار مهم است.

گزارش قبلی موارد زیر را نشان داده است:

- 4,678 Cross-Split Duplicates
- هشدار Scene Leakage
- تعداد بسیار زیاد Duplicate Samples

### ۵.۱. تقسیم‌بندی بر اساس Scene

در حالت Real، تقسیم Train و Validation باید تا حد امکان بر اساس Scene انجام شود.

هیچ Scene مشترکی نباید در Train و Validation وجود داشته باشد.

اگر split رسمی nuScenes Prediction برای مسئله موردنظر قابل استفاده است، ابتدا اعتبار و سازگاری آن را با Task فعلی بررسی کن.

در غیر این صورت، از یک تقسیم‌بندی Scene-level قطعی و مستند استفاده کن.

### ۵.۲. شناسه‌های یکتا

برای هر نمونه حداقل موارد زیر را ذخیره کن:

- sample_token
- scene_token
- instance_token
- timestamp
- category

شناسه یکتای یک نمونه باید ترکیبی از اطلاعات لازم برای شناسایی یک مشاهده از یک عامل در یک زمان مشخص باشد.

صرفاً استفاده از instance_token برای تشخیص Duplicate صحیح نیست؛ زیرا یک عامل می‌تواند در زمان‌های مختلف چندین مشاهده معتبر داشته باشد.

### ۵.۳. Duplicate Detection

سه نوع تکرار را جداگانه بررسی کن:

1. Duplicate Sample ID
2. Duplicate Agent-Time Observation
3. Duplicate Trajectory Content

تکرار Scene ID را به‌عنوان Duplicate Sample گزارش نکن.

قبل از حذف هر نوع تکرار، دلیل و معیار تشخیص را مشخص کن.

از حذف مسیرهای معتبر صرفاً به علت شباهت هندسی خودداری کن.

### ۵.۴. Split Validation

پس از استخراج، بررسی کن:

- Train Scene IDs ∩ Validation Scene IDs = Empty
- Train Sample IDs ∩ Validation Sample IDs = Empty
- Train Agent-Time IDs ∩ Validation Agent-Time IDs = Empty

در صورت مشاهده نشت داده، وضعیت Dataset را WARNING یا FAILED تعیین کن.

---

## ۶. اصلاح انتخاب Target Categories

در گزارش قبلی دسته‌هایی مانند خودرو، عابر پیاده، مانع، Traffic Cone و Debris همگی به‌عنوان Target مشاهده شده‌اند.

برای پیش‌بینی مسیر، دسته‌های هدف باید آگاهانه انتخاب شوند.

پیکربندی Category Filtering را توسعه بده.

حداقل از دسته‌های زیر پشتیبانی کن:

### Vehicle Categories

- vehicle.car
- vehicle.truck
- vehicle.bus.rigid
- vehicle.bus.bendy
- vehicle.trailer
- vehicle.motorcycle
- vehicle.bicycle

### Pedestrian Categories

- human.pedestrian.adult
- human.pedestrian.child
- human.pedestrian.construction_worker
- human.pedestrian.police_officer
- human.pedestrian.personal_mobility
- human.pedestrian.stroller
- human.pedestrian.wheelchair

### سایر دسته‌ها

دسته‌های ثابت یا غیرمتحرک را به‌صورت پیش‌فرض از Target Agentها جدا کن، مگر اینکه صریحاً برای Task انتخاب شوند.

امکان تنظیم موارد زیر وجود داشته باشد:

- Target Categories
- Neighbor Categories
- Excluded Categories
- Minimum Category Samples
- Category Balancing Strategy

توجه کن که دسته‌بندی Agent هدف و Neighbor نباید الزاماً یکسان باشد.

در گزارش نهایی تعداد و درصد هر دسته در Train و Validation ثبت شود.

---

## ۷. اصلاح Sampling و تعداد نمونه‌های استخراجی

در نسخه فعلی پارامتر `target_count` می‌تواند باعث محدود شدن تعداد نمونه‌ها شود.

این رفتار را دقیق‌تر و شفاف‌تر کن.

حالت‌های زیر پشتیبانی شوند:

- Extract All
- Fixed Sample Count
- Percentage-based Sampling
- Scene-based Sampling
- Category-aware Sampling

الزامات:

1. در حالت All، تمام نمونه‌های معتبر استخراج شوند.
2. در حالت Fixed، تعداد نمونه‌ها دقیقاً تا سقف درخواست‌شده محدود شود.
3. در حالت Percentage، درصد واقعی استخراج گزارش شود.
4. Sampling باید قابل بازتولید باشد.
5. Sampling نباید باعث حذف ناخواسته دسته‌های کم‌نمونه شود.
6. در صورت استفاده از Stratified Sampling، تعداد هر دسته ثبت شود.
7. تعداد واقعی نمونه‌های استخراج‌شده با تعداد درخواستی مقایسه شود.
8. هیچ عددی در گزارش بدون محاسبه از داده واقعی تولید نشود.

پارامتر Seed باید تمام مراحل تصادفی را کنترل کند.

---

## ۸. اصلاح Synthetic Fallback

در نسخه فعلی، حالت `auto` ممکن است در صورت نبود Map یا شکست استخراج واقعی به Synthetic Generator منتقل شود.

این رفتار برای آزمایش‌های پژوهشی خطرناک است.

سیاست جدید:

### Real Mode

- فقط داده واقعی nuScenes مجاز است.
- در صورت شکست، Exception ثبت شود.
- هیچ Synthetic Fallback انجام نشود.
- Dataset به‌عنوان FAILED معرفی شود.

### Synthetic Mode

- فقط در صورت درخواست صریح کاربر اجرا شود.
- Synthetic بودن تمام نمونه‌ها مشخص باشد.
- خروجی در پوشه‌ای مستقل ذخیره شود.

### Auto Mode

- ابتدا اعتبار فایل‌های واقعی و قابلیت بارگذاری SDK بررسی شود.
- در صورت موفقیت، استخراج Real انجام شود.
- در صورت شکست، به‌صورت پیش‌فرض متوقف شود.
- Fallback به Synthetic فقط با یک پارامتر صریح مانند `--allow-synthetic-fallback` مجاز باشد.

در حالت Real، فقدان HD Map نباید موجب تولید داده مصنوعی شود؛ بلکه باید نقص Map به‌صورت مستقل گزارش شود.

---

## ۹. اعتبارسنجی کیفیت داده‌ها

قبل از ذخیره‌سازی، تمام نمونه‌ها را بررسی کن.

موارد زیر باید بررسی شوند:

- NaN
- Inf
- Invalid coordinates
- Empty past trajectory
- Empty future trajectory
- Incorrect trajectory length
- Invalid timestamps
- Duplicate timestamps
- Non-monotonic timestamps
- Invalid neighbor masks
- Invalid map polylines
- Invalid category
- Missing instance token
- Missing sample token
- Missing scene token
- Coordinate-system mismatch

برای هر مشکل:

- تعداد موارد
- درصد از کل
- نام Field
- علت
- سیاست برخورد

ثبت شود.

هیچ نمونه‌ای را بی‌صدا حذف نکن.

صفر بودن مختصات به‌تنهایی نباید Invalid تلقی شود.

داده Missing را از داده Invalid تفکیک کن.

---

## ۱۰. استانداردسازی ساختار خروجی Dataset

ساختار خروجی باید با Dataset Loader فعلی پروژه سازگار باشد.

ابتدا ساختار مورد انتظار مدل و DataLoader را استخراج کن و سپس خروجی را بر اساس آن طراحی کن.

در صورت امکان، ساختار پیشنهادی زیر را در نظر بگیر:

`sample_id`

`sample_token`

`instance_token`

`scene_token`

`scene_name`

`timestamp`

`city`

`category`

`past_traj`

`future_traj`

`neighbors_past`

`neighbors_mask`

`neighbor_categories`

`map_lanes`

`map_lane_types`

`map_lane_mask`

`coordinate_frame`

`sampling_interval`

`metadata`

اگر تغییر نام یا اضافه کردن فیلد باعث ناسازگاری با مدل فعلی می‌شود، یک لایه سازگاری ایجاد کن یا فرمت قدیمی را حفظ کن.

تمام اطلاعات ضروری برای تشخیص و اعتبارسنجی نمونه باید ذخیره شوند.

---

## ۱۱. توسعه Dataset Writer

کلاس `SequenceDatasetWriter` را اصلاح کن.

الزامات:

1. خروجی Train و Validation جدا باشد.
2. ذخیره‌سازی با JSON معتبر انجام شود.
3. فایل‌های بزرگ به‌صورت Streaming یا با روشی کم‌مصرف در حافظه ذخیره شوند.
4. ذخیره‌سازی اتمیک انجام شود تا در صورت قطع اجرا فایل ناقص جایگزین فایل سالم نشود.
5. فایل‌های قبلی بدون تأیید کاربر بازنویسی نشوند.
6. ترتیب نمونه‌ها به‌صورت قطعی و قابل بازتولید باشد.
7. اطلاعات Metadata و Configuration ذخیره شوند.
8. آمار نهایی هر Split ثبت شود.

در صورت نیاز، فرمت ذخیره‌سازی کارآمدتری مانند NPZ یا Parquet را بررسی کن؛ اما تغییر فرمت را فقط پس از بررسی سازگاری Dataset Loader انجام بده.

برای حجم زیاد داده، از نگهداری غیرضروری تمام نمونه‌ها در RAM اجتناب کن.

---

## ۱۲. توسعه گزارش خروجی

پس از آماده‌سازی دیتاست، گزارش مستقل تولید شود.

اطلاعات گزارش شامل:

- Dataset Type
- Dataset Version
- Dataset Root
- Extraction Mode
- Real / Synthetic Counts
- Total Samples
- Train Samples
- Validation Samples
- Scene Counts
- Category Distribution
- Temporal Availability
- Interaction Availability
- HD Map Availability
- Neighbor Statistics
- Lane Statistics
- Invalid Samples
- Duplicate Samples
- Split Leakage
- Sampling Configuration
- Extraction Duration
- Failed Samples
- Filtered Samples
- Warnings

آمار باید از داده‌های واقعی استخراج‌شده محاسبه شود.

گزارش باید با `data_summery.py` قابل مقایسه باشد.

اگر آمار استخراج با آمار گزارش مستقل متفاوت بود، علت اختلاف باید قابل پیگیری باشد.

---

## ۱۳. توسعه معماری و کیفیت کد

اصول SOLID فعلی را حفظ کن، اما اولویت را به صحت داده و سادگی نگهداری بده.

کلاس‌های موجود را در صورت نیاز توسعه بده:

- `NuScenesMapExpansionValidator`
- `NuScenesRealDataExtractor`
- `SyntheticTrajectoryGenerator`
- `SequenceDatasetWriter`
- `DataPreparationPipeline`

در صورت نیاز، کلاس‌های مستقل برای موارد زیر اضافه کن:

- Coordinate Transformer
- Trajectory Validator
- Neighbor Extractor
- HD Map Extractor
- Split Validator
- Dataset Quality Analyzer

از توابع طولانی و مسئولیت‌های متعدد اجتناب کن.

برای تمام مراحل مهم Logging مناسب اضافه کن.

Exceptionها را بی‌صدا نادیده نگیر.

به‌خصوص در استخراج HD Map، عبارت‌هایی مانند `except Exception: pass` را حذف یا اصلاح کن تا علت شکست قابل تشخیص باشد.

---

## ۱۴. CLI و پیکربندی

پارامترهای قابل تنظیم را با argparse ارائه کن.

پارامترهای موجود را حفظ کن و در صورت نیاز گزینه‌های زیر را اضافه کن:

- `--dataroot`
- `--output_dir`
- `--mode`
- `--count`
- `--category`
- `--seed`
- `--max_train_scenes`
- `--max_val_scenes`
- `--sampling_strategy`
- `--train_ratio`
- `--map_radius`
- `--max_lanes`
- `--max_neighbors`
- `--neighbor_radius`
- `--allow-synthetic-fallback`
- `--validate-only`
- `--overwrite`

مقادیر پیش‌فرض باید از `DataConfig` دریافت شوند، مگر آنکه بررسی پروژه نشان دهد مقدار پیش‌فرض نادرست یا ناسازگار است.

از hardcode کردن مسیرهای مخصوص یک سیستم‌عامل یا کامپیوتر خودداری کن.

---

## ۱۵. تست و اعتبارسنجی

پس از پیاده‌سازی، تست‌های زیر را انجام بده:

### Real Dataset Tests

- بارگذاری واقعی nuScenes
- استخراج Past و Future
- بررسی نرخ 10Hz
- استخراج Neighborها
- استخراج HD Map از هر چهار شهر
- بررسی سازگاری دستگاه مختصات
- بررسی شناسه‌های نمونه
- بررسی Scene-level split

### Edge Cases

- نبود Map برای یک شهر
- نبود Neighbor
- نبود Future trajectory
- داده ناقص
- NaN / Inf
- Timestamp نامعتبر
- Duplicate sample
- Category نامعتبر
- Map polyline ناقص
- شکست SDK

### Pipeline Tests

- Real Mode
- Synthetic Mode
- Auto Mode
- Sampling
- ذخیره‌سازی Train و Validation
- بارگذاری خروجی با Dataset Loader واقعی پروژه
- سازگاری با DARTS Searcher
- اجرای مستقل از main.py

### Acceptance Tests

در پایان حتماً بررسی کن:

1. خروجی واقعی فاقد داده مصنوعی باشد.
2. HD Map در صورت وجود داده و SDK معتبر واقعاً استخراج شود.
3. Train و Validation هیچ Scene مشترکی نداشته باشند.
4. Duplicate Sample ID در هر Split صفر باشد.
5. همسایگان بر اساس فاصله واقعی انتخاب شوند.
6. Maskها با ابعاد داده سازگار باشند.
7. خروجی با Dataset Loader موجود قابل بارگذاری باشد.
8. اجرای گزارش `data_summery.py` بعد از آماده‌سازی، آمار معنادار و سازگار ارائه دهد.

اگر دیتاست واقعی در محیط توسعه در دسترس نیست، تست واقعی را موفق اعلام نکن. تست‌های ساختگی فقط برای بررسی منطق نرم‌افزار مجازند و باید از تست‌های Real جداگانه گزارش شوند.

---

## ۱۶. محدودیت‌های تغییرات

- فایل اصلی هدف `python_src/data/prepare_data.py` است.
- `main.py` را برای اجرای این ابزار تغییر نده.
- به منطق DARTS دست نزن، مگر برای رفع ناسازگاری اثبات‌شده در قرارداد داده.
- مدل نهایی را بدون ضرورت تغییر نده.
- هیچ داده مصنوعی را به‌عنوان داده واقعی معرفی نکن.
- هیچ خطای استخراجی را بی‌صدا نادیده نگیر.
- هیچ آمار یا نتیجه آزمایشی را جعل نکن.
- تغییرات سایر فایل‌ها باید حداقلی، ضروری و مستند باشند.

---

## ۱۷. خروجی نهایی مورد انتظار

در پایان کار ارائه کن:

1. نسخه کامل و اصلاح‌شده `prepare_data.py`.
2. شرح تغییرات هر کلاس و متد.
3. فهرست مشکلات نسخه قبلی که برطرف شده‌اند.
4. فهرست فایل‌های دیگری که تغییر کرده‌اند و دلیل هر تغییر.
5. دستورات دقیق اجرای استخراج Real و Synthetic.
6. ساختار نهایی فایل‌های خروجی.
7. نتیجه تست‌های واقعی و مصنوعی به‌صورت تفکیک‌شده.
8. میزان پوشش واقعی HD Map در هر شهر.
9. آمار نهایی Neighbors.
10. تعداد نمونه‌های Train و Validation.
11. نتیجه بررسی Duplicate و Scene Leakage.
12. نتیجه سازگاری با Dataset Loader و DARTS Searcher.
13. محدودیت‌های باقی‌مانده.

**معیار نهایی موفقیت:**

خروجی `prepare_data.py` باید یک دیتاست واقعی، معتبر، بدون نشت بین Train و Validation، دارای اطلاعات زمانی صحیح، همسایگان معتبر و HD Map واقعی (در صورت دسترسی به Map Assets) باشد که بدون تغییر غیرضروری در ساختار فعلی پروژه، برای جستجوی معماری و آموزش مدل TrajectoryLNAS قابل استفاده باشد.

اولویت مطلق با صحت داده‌ها و اعتبار علمی آزمایش‌ها است، نه صرفاً افزایش تعداد نمونه‌های استخراج‌شده.