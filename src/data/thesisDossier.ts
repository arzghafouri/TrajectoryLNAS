export interface ThesisMetadata {
  docNumber: string;
  irandocNumber: string;
  researchNumber: string;
  proposalDate: string;
  titleFa: string;
  titleEn: string;
  keywordsFa: string[];
  keywordsEn: string[];
  thesisType: string;
  duration: string;
  credits: number;
  student: {
    name: string;
    studentId: string;
    entranceYear: string;
    semester: string;
    major: string;
    phone: string;
    email: string;
  };
  supervisors: {
    name: string;
    role: string;
    degree: string;
    field: string;
    department: string;
    university: string;
    phone: string;
    email: string;
  }[];
  budget: {
    row: number;
    type: string;
    title: string;
    costRials: string;
  }[];
  totalBudgetRials: string;
}

export const THESIS_DATA: ThesisMetadata = {
  docNumber: 'کاربرگ شماره ۲',
  irandocNumber: '۱۹۸۷۵۶۸',
  researchNumber: '۲۸۲۴۵۲۵۹',
  proposalDate: '۱۴۰۵/۰۳/۰۷',
  titleFa: 'بهینه‌سازی معماری پیش‌بینی مسیر با استفاده از روش جستجوی معماری شبکه عصبی سبک',
  titleEn: 'TrajectoryLNAS: Trajectory Prediction Architecture Optimization via Lightweight Neural Architecture Search',
  keywordsFa: [
    'پیش‌بینی مسیر',
    'جستجوی معماری شبکه عصبی (NAS)',
    'رانندگی خودران',
    'ارزیابی پروکسی بدون هزینه',
    'شبکه‌ی عصبی فراگیر (Super-net)',
    'جستجوی معماری مشتق‌پذیر (DARTS)'
  ],
  keywordsEn: [
    'Trajectory Prediction',
    'Neural Architecture Search (NAS)',
    'Autonomous Driving',
    'Zero-Cost Proxy Evaluation',
    'Super-net',
    'Differentiable Architecture Search (DARTS)'
  ],
  thesisType: 'توسعه‌ای',
  duration: 'شش ماهه',
  credits: 6,
  student: {
    name: 'علیرضا غفوری',
    studentId: '۴۰۳۱۶۳۱۲۰۹',
    entranceYear: '۱۴۰۳',
    semester: 'نیمسال اول',
    major: 'مهندسی کامپیوتر – هوش مصنوعی و رباتیک',
    phone: '۰۹۳۶۲۴۰۴۳۰۸',
    email: 'alirezaghafouri@cuir.ac.ir',
  },
  supervisors: [
    {
      name: 'دکتر زینب قاسمی نراقی',
      role: 'استاد راهنما',
      degree: 'دکتری',
      field: 'مهندسی کامپیوتر - هوش مصنوعی',
      department: 'مهندسی کامپیوتر',
      university: 'دانشگاه خواجه نصیرالدین طوسی',
      phone: '۰۹۱۲۱۷۱۶۸۱۳',
      email: 'zeinabnaraaghi@gmail.com',
    },
    {
      name: 'دکتر زینب عابدینی',
      role: 'استاد مشاور',
      degree: 'دکتری',
      field: 'مهندسی کامپیوتر - هوش مصنوعی',
      department: 'مهندسی کامپیوتر',
      university: 'دانشگاه صنعتی شریف',
      phone: '۰۹۱۲۷۵۰۱۷۰۰',
      email: 'z.abedini@sharif.edu',
    },
  ],
  budget: [
    {
      row: 1,
      type: 'مواد و وسایل مصرفی',
      title: 'ـ',
      costRials: 'ـ',
    },
    {
      row: 2,
      type: 'مواد و وسایل غیرمصرفی',
      title: 'تهیه سیستم مناسب با کارت گرافیک میان‌رده',
      costRials: '۲,۰۰۰,۰۰۰,۰۰۰',
    },
    {
      row: 3,
      type: 'آزمایشات و خدمات تخصصی',
      title: 'ـ',
      costRials: 'ـ',
    },
    {
      row: 4,
      type: 'سایر هزینه‌ها',
      title: 'شامل خرید کتاب‌ها، مقالات یا اشتراک سرویس‌های آنلاین و هزینه‌های پیش‌بینی نشده',
      costRials: '۲۰۰,۰۰۰,۰۰۰',
    },
  ],
  totalBudgetRials: '۲,۲۰۰,۰۰۰,۰۰۰ ریال (معادل ۲۲۰ میلیون تومان)',
};

export const THESIS_SECTIONS = [
  {
    id: '4-1',
    title: '۴-۱. بیان مسئله (Problem Statement)',
    contentFa: `سیستم‌های پیش‌بینی مسیر یکی از مؤلفه‌های کلیدی در سامانه‌های رانندگی خودکار محسوب می‌شوند؛ زیرا تصمیم‌گیری ایمن خودرو وابسته به درک رفتار آینده سایر خودروها، عابران و دوچرخه‌سواران است. خطا در پیش‌بینی مسیر می‌تواند منجر به انتخاب مانورهای نامناسب و در نهایت افزایش احتمال تصادف شود.

در سال‌های اخیر، مدل‌های یادگیری عمیق از جمله RNN، CNN، معماری‌های مبتنی بر ترنسفورمر و مدل‌های گرافی نظیر LaneGCN توانسته‌اند با مدل‌سازی تعامل میان چند عامل، دقت پیش‌بینی را بهبود دهند. با این حال طراحی معماری این شبکه‌ها عمدتاً مبتنی بر آزمون و خطا و وابسته به تجربه پژوهشگر بوده است.

روش‌های پیشین مانند TrajectoryNAS با بهره‌گیری از الگوریتم‌های فراابتکاری (نظیر تبرید شبیه‌سازی‌شده چندهدفه MOSA) اگرچه به معماری‌های مناسبی دست می‌یابند، اما به دلیل نیاز به ارزیابی صدها معماری و آموزش کامل آن‌ها، نیازمند روزها تا هفته‌ها محاسبات پردازنده‌های گرافیکی گران‌قیمت هستند.
مسئله اصلی این پژوهش آن است که چگونه می‌توان هزینه محاسباتی و زمان جستجوی معماری در روش‌های NAS برای پیش‌بینی مسیر را به طور قابل‌توجهی کاهش داد، بدون آنکه دقت پیش‌بینی نسبت به روش‌های موجود افت کند.`,
  },
  {
    id: '4-2',
    title: '۴-۲. اهمیت و ضرورت (Significance & Necessity)',
    contentFa: `۱. ایمنی مسافران و رانندگان در گرو تصمیم‌گیری دقیق و به‌موقع برای حرکت خودرو است.
۲. طراحی دستی معماری‌ها نیازمند آزمون و خطاهای متعدد و زمان توسعه طولانی است.
۳. روش‌های فراابتکاری به دلیل نیاز به سخت‌افزارهای سنگین کلاستر GPU، دسترسی بسیاری از پژوهشگران و صنایع را محدود می‌کنند.
۴. سامانه‌های خودران در لبه (Edge Computing) منابع محدودی از نظر توان پردازشی و مصرف انرژی دارند؛ بنابراین ساختار کشف‌شده باید سبک، کم‌تأخیر و با حداقل FLOPs باشد.`,
  },
  {
    id: '4-3',
    title: '۴-۳. اهداف پژوهش (Research Objectives)',
    contentFa: `• توسعه روشی برای جستجوی معماری مشتق‌پذیر (Differentiable NAS) و بهینه در حوزه پیش‌بینی مسیر.
• طراحی فضای جستجوی اختصاصی با عملگرهای سه‌گانه متناسب با داده‌های مکانی-زمانی خودرو (Temporal, Interaction, HD Map).
• به‌کارگیری ارزیابی‌های پروکسی بدون هزینه (Zero-Cost Proxies نظیر SNIP و SynFlow) به منظور رتبه‌بندی سریع شبکه‌ها بدون نیاز به آموزش کامل.
• ارزیابی دقیق با معیارهای استاندارد ADE (خطای میانگین جابه‌جایی) و FDE (خطای نهایی جابه‌جایی) همراه با سنجش تأخیر نظری و عملیات ممیز شناور (FLOPs).`,
  },
  {
    id: '4-4',
    title: '۴-۴. سوالات پژوهش (Research Questions)',
    contentFa: `۱. چگونه می‌توان یک روش جستجوی سبک (Lightweight NAS) طراحی کرد که هم‌زمان زمان جستجو را کاهش و دقت را حفظ نماید؟
۲. راهبرد جستجوی مشتق‌پذیر (DARTS) در مقایسه با الگوریتم‌های فراابتکاری تکاملی از منظر سرعت همگرایی و مصرف منابع چه اثربخشی دارد؟
۳. تا چه اندازه معیارهای پروکسی بدون هزینه می‌توانند به عنوان جایگزین آموزش کامل مدل در فاز جستجو عمل کنند؟
۴. ویژگی‌های ساختاری معماری استخراج‌شده چه تفاوتی با مدل‌های طراحی‌شده دستی دارد؟`,
  },
  {
    id: '4-5',
    title: '۴-۵. فرضیه‌های پژوهش (Hypotheses)',
    contentFa: `• فرض ۱: جستجوی مشتق‌پذیر و ارزیابی پروکسی امکان دستیابی به دقتی بسیار نزدیک به نسخه اصلی TrajectoryNAS را با کاهش چشمگیر زمان جستجو فراهم می‌کند.
• فرض ۲: همبستگی بالایی (Correlation Coefficient) میان پروکسی‌های بدون هزینه (مانند حساسیت گرادیان SNIP و جریان سیناپسی SynFlow) و عملکرد نهایی مدل وجود دارد.
• فرض ۳: هرس و حذف هوشمندانه بخش‌های غیرضروری از فضای جستجو، اثر منفی بر دقت نهایی مدل نخواهد گذاشت.`,
  },
  {
    id: '4-7',
    title: '۴-۷. جنبه‌های نوآوری طرح (Innovations)',
    contentFa: `۱. جایگزینی الگوریتم فراابتکاری MOSA با بهینه‌سازی گرادیانی مشتق‌پذیر DARTS (سرعت همگرایی فوق‌العاده بالا).
۲. استفاده از معیارهای سخت‌افزاری و پروکسی‌های بدون هزینه (FLOPs & Zero-Cost Proxies) در تابع هزینه معماری.
۳. تعریف فضای جستجوی سه‌لایه ویژه پویایی خودرو:
   - لایه زمانی: GRU, LSTM, TCN, Temporal Attention
   - لایه تعاملی: GCN, GAT, Social Pooling, Pairwise Attention
   - لایه نقشه: Lane Attention, Lane Graph, Map Feature Fusion
۴. استفاده از توابع خطای جابه‌جایی ADE و FDE به جای Cross-Entropy متداول در مسائل طبقه‌بندی تصویر.`,
  },
  {
    id: '4-8',
    title: '۴-۸. روش انجام پژوهش و مراحل پیاده‌سازی (Methodology)',
    contentFa: `۱. انتخاب و اعتبارسنجی روی دیتاست‌های استاندارد رانندگی خودران nuScenes و Argoverse.
۲. طراحی فضای جستجوی سبک و تعریف عملگرهای تخصصی.
۳. پیاده‌سازی معماری فراگیر (Super-net) مبتنی بر اشتراک وزن‌ها (Weight Sharing).
۴. بهینه‌سازی دوسطحی مشتق‌پذیر با تفکیک پارامترهای وزن (w) و پارامترهای معماری (alpha).
۵. ارزیابی مبتنی بر پروکسی‌های بدون هزینه و جریمه سخت‌افزاری FLOPs در تابع هزینه.
۶. استخراج و گسسته‌سازی معماری نهایی با انتخاب عملگر با بالاترین مقدار alpha.
۷. آموزش نهایی مدل استخراج‌شده از صفر (Standalone Training).
۸. ارزیابی جامع با معیارهای ADE, FDE, Miss Rate, Latency و مقایسه با TrajectoryNAS اصلی.`,
  },
  {
    id: '4-9',
    title: '۴-۹. دستاوردها و کاربردها (Outcomes & Applications)',
    contentFa: `• کاهش ۳۰ تا ۶۰ درصدی (و در تست‌های اولیه تا ۹۰٪) زمان و هزینه جستجو نسبت به روش‌های تکاملی.
• ارائه چارچوب جامع متن‌باز برای طراحی خودکار شبکه‌های پیش‌بینی حرکت در سیستم‌های لبه‌ای خودران.
• فراهم‌سازی امکان استقرار بلادرنگ روی بردهای نهفته خودرو (مانند سری Jetson و DRIVE AGX).`,
  },
];
