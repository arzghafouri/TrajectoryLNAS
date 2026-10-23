import React, { useState } from 'react';
import {
  BookOpen,
  Code2,
  Cpu,
  Layers,
  Sparkles,
  Terminal,
  Zap,
  ArrowRight,
  CheckCircle2,
  Copy,
  Check,
  Workflow,
  HelpCircle,
  FileCode,
  FolderTree,
  ChevronDown,
  ChevronUp,
  GraduationCap,
  Play,
  Flame,
  Search,
  ExternalLink
} from 'lucide-react';

interface MethodLesson {
  methodName: string;
  signature: string;
  returnType: string;
  location: string;
  difficulty: 'مقدماتی' | 'متوسط' | 'پیشرفته';
  stepGoal: string;
  inputsExplained: { param: string; type: string; desc: string }[];
  lineByLineTeaching: { line: string; explanation: string }[];
  commonJuniorMistake: string;
}

interface ClassCourse {
  classId: string;
  className: string;
  fileLocation: string;
  inheritance: string;
  roleBadge: string;
  classPurpose: string;
  constructorExplained: {
    signature: string;
    description: string;
    params: { name: string; desc: string }[];
  };
  methods: MethodLesson[];
}

export const JuniorDevDocumentation: React.FC = () => {
  const [selectedClassId, setSelectedClassId] = useState<string>('DARTSSearcher');
  const [selectedMethodIdx, setSelectedMethodIdx] = useState<number>(0);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');

  const handleCopy = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const curriculum: ClassCourse[] = [
    {
      classId: 'DARTSSearcher',
      className: 'class DARTSSearcher(BaseTrainer)',
      fileLocation: 'python_src/core/darts_searcher.py',
      inheritance: 'BaseTrainer -> nn.Module',
      roleBadge: 'هسته بهینه‌سازی دوسطحی (Stage 1)',
      classPurpose: 'مسئول اصلی موتور جستجوی معماری مشتق‌پذیر. این کلاس با بهینه‌سازی هم‌زمان وزن‌های شبکه (w) و ضرایب احتمال انتخاب عملگرها (α) با در نظر گرفتن جریمه پیوسته FLOPs، بهترین ساختار عصبی را کشف می‌کند.',
      constructorExplained: {
        signature: 'def __init__(self, search_config: SearchConfig, model_config: Optional[ModelConfig] = None):',
        description: 'مقداردهی اولیه سوپرنت، ایجاد دو اپتیمایزر مستقل (یکی برای وزن‌ها و یکی برای آلفاها)، و راه‌اندازی دیتالودرها.',
        params: [
          { name: 'search_config', desc: 'هایپرپارامترهای بهینه‌سازی دوسطحی (تعداد اپوک‌ها، نرخ‌های یادگیری lr_w و lr_a، ضریب جریمه لاندای فلوپس)' },
          { name: 'model_config', desc: 'ابعاد پایه مدل (طول گذشته ۲۰ گام، آینده ۳۰ گام، d_model=128)' }
        ]
      },
      methods: [
        {
          methodName: 'train_epoch(self, epoch: int)',
          signature: 'def train_epoch(self, epoch: int) -> Dict[str, float]:',
          returnType: 'Dict[str, float]: {"train_w_loss": float, "train_a_loss": float}',
          location: 'core/darts_searcher.py:75',
          difficulty: 'پیشرفته',
          stepGoal: 'اجرای یک اپوک از بهینه‌سازی دوسطحی DARTS (آپدیت متناوب وزن‌ها روی دیتای Train و سپس آپدیت آلفاها روی دیتای Val).',
          inputsExplained: [
            { param: 'epoch', type: 'int', desc: 'شماره اپوک جاری جهت محاسبه شیب دمای سافت‌مکس (Temperature Annealing)' }
          ],
          lineByLineTeaching: [
            {
              line: 'temp = self.get_temperature(epoch)',
              explanation: 'گام ۱: محاسبه دمای جاری. در اپوک ۱ دما ۰٫۹۵ است (کاوش آزاد بین تمام عملگرها). با نزدیک شدن به اپوک ۳۰ دما به ۰٫۲ می‌رسد تا توزیع تیز شده و یک عملگر قاطعانه برنده شود.'
            },
            {
              line: 'self.optimizer_w.zero_grad(); out = self.supernet(past, neighbors, map_lanes, temp)',
              explanation: 'گام ۲: اجرای Forward Pass روی بچ داده‌های آموزشی با استفاده از بردار احتمالات وزن‌دار عملگرها (MixedOps).'
            },
            {
              line: 'loss_w = self.criterion(out, future); loss_w.backward()',
              explanation: 'گام ۳: محاسبه خطای L2 بین مسیر پیش‌بینی‌شده و واقعی خودروها و انتشار پس‌رو برای آپدیت وزن‌های کانولوشن‌ها و گیت‌ها.'
            },
            {
              line: 'val_batch = next(self.val_iter); loss_alpha = self.compute_arch_loss(val_batch, temp)',
              explanation: 'گام ۴ (شاهکار DARTS): گرادیان‌گیری از آلفاها روی داده‌های اعتبارسنجی (Validation) به همراه جریمه پیوسته FLOPs تا مدلی انتخاب شود که تعمیم‌پذیر و سریع باشد.'
            },
            {
              line: 'self.optimizer_a.step()',
              explanation: 'گام ۵: به‌روزرسانی پارامترهای معماری به سمت عملگرهایی که خطای کمتر و سرعت بالاتری داشتند.'
            }
          ],
          commonJuniorMistake: 'اشتباه متداول جونیورها این است که آلفاها را هم با داده آموزشی آپدیت می‌کنند! اگر آلفا با Train آپدیت شود، مدل سنگین‌ترین عملگر را صرفاً حفظ (Overfit) می‌کند، نه بهترین را!'
        },
        {
          methodName: 'discretize_architecture(self)',
          signature: 'def discretize_architecture(self) -> Tuple[int, int, int]:',
          returnType: 'Tuple[int, int, int]: (best_temporal_idx, best_inter_idx, best_map_idx)',
          location: 'core/darts_searcher.py:215',
          difficulty: 'متوسط',
          stepGoal: 'تبدیل سوپرنت پیوسته به یک مدل قطعی (Discrete) سبک با برگزیدن عملگری که بیشترین احتمال آلفا را به دست آورده است.',
          inputsExplained: [],
          lineByLineTeaching: [
            {
              line: 'a_temp, a_inter, a_map = self.arch_params.softmax_all()',
              explanation: 'گام ۱: اعمال سافت‌مکس روی هر سه بردار آلفا تا مقادیر به صورت درصد احتمال تفکیک شوند.'
            },
            {
              line: 'best_t = torch.argmax(a_temp).item()',
              explanation: 'گام ۲: پیدا کردن ایندکس عملگر برنده زمانی (مثلاً ایندکس ۳ که TemporalConvEncoder است).'
            },
            {
              line: 'best_i = torch.argmax(a_inter).item()',
              explanation: 'گام ۳: پیدا کردن ایندکس عملگر برنده تعاملی (مثلاً ایندکس ۳ که SocialPooling است).'
            },
            {
              line: 'best_m = torch.argmax(a_map).item()',
              explanation: 'گام ۴: پیدا کردن ایندکس عملگر نقشه HD (مثلاً ایندکس ۳ که MapFeatureFusion است).'
            },
            {
              line: 'return best_t, best_i, best_m',
              explanation: 'گام ۵: بازگرداندن این تاپل سه‌تایی برای تحویل به فاز دوم (آموزش مدل نهایی از صفر).'
            }
          ],
          commonJuniorMistake: 'استفاده از مقادیر خام لاجیت‌ها به جای سافت‌مکس؛ باید حتماً نرمال‌سازی با دمای مناسب انجام شده باشد.'
        },
        {
          methodName: 'profile_zero_cost_proxies(self)',
          signature: 'def profile_zero_cost_proxies(self) -> Dict[str, float]:',
          returnType: 'Dict[str, float]: {"snip": float, "synflow": float}',
          location: 'core/darts_searcher.py:120',
          difficulty: 'پیشرفته',
          stepGoal: 'سنجش سلامت انتقال گرادیان و پتانسیل یادگیری شبکه در اپوک صفر، بدون صرف زمان آموزش.',
          inputsExplained: [],
          lineByLineTeaching: [
            {
              line: 'snip_val = snip_score(self.supernet, sample_batch)',
              explanation: 'محاسبه حساسیت گرادیان نسبت به پارامترها با یک مرحله Forward/Backward ساده روی یک مینی‌بچ.'
            },
            {
              line: 'synflow_val = synflow_score(self.supernet, input_shapes)',
              explanation: 'تزریق بردار سراسر ۱ (All-ones) و ضرب تانسوری گرادیان در وزن‌ها جهت اطمینان از عدم وقوع فروپاشی لایه‌ها (Layer Collapse).'
            }
          ],
          commonJuniorMistake: 'فراموش کردن zero_grad قبل از محاسبه پروکسی‌ها که باعث جمع‌شدن گرادیان‌های قبلی می‌شود.'
        }
      ]
    },
    {
      classId: 'FinalModelTrainer',
      className: 'class FinalModelTrainer(BaseTrainer)',
      fileLocation: 'python_src/core/final_trainer.py',
      inheritance: 'BaseTrainer -> nn.Module',
      roleBadge: 'آموزش مستقل مدل نهایی با Early Stopping (Stage 2)',
      classPurpose: 'پس از کشف ۳ عملگر برتر، این کلاس مدل سبک نهایی (FinalTrajectoryModel) را از صفر ساخته و با نظارت دقیق آستانه صبر، بهترین وزن‌ها را روی دیسک ذخیره می‌کند.',
      constructorExplained: {
        signature: 'def __init__(self, best_op_indices: Tuple[int, int, int], train_config: TrainConfig, model_config: Optional[ModelConfig] = None):',
        description: 'ساخت مدل اختصاصی با عملگرهای کشف‌شده و تنظیم پارامترهای توقف زودهنگام (Patience و Min Delta).',
        params: [
          { name: 'best_op_indices', desc: 'تاپل سه‌تایی عملگرهای منتخب حاصل از فاز جستجو (مثلاً 3, 3, 3)' },
          { name: 'train_config', desc: 'تنظیمات آموزش شامل سقف ۵۰ اپوک، patience=8، lr=1e-3 و مسیر save_path' }
        ]
      },
      methods: [
        {
          methodName: 'train_epoch(self, epoch: int) [Physics-Informed Loss & GradAccum]',
          signature: 'def train_epoch(self, epoch: int) -> float:',
          returnType: 'float: میانگین تابع هزینه ترکیبی (Huber + ADE/FDE + Physics)',
          location: 'core/final_trainer.py:73',
          difficulty: 'پیشرفته',
          stepGoal: 'آموزش گام‌به‌گام با تابع اتلاف آگاه از فیزیک حرکت (Physics-Informed Loss) و تجمیع گرادیان ۴ گامه (Effective Batch=128).',
          inputsExplained: [
            { param: 'epoch', type: 'int', desc: 'شماره اپوک جهت اعمال زمان‌بندی نرخ یادگیری کسینوسی' }
          ],
          lineByLineTeaching: [
            {
              line: 'loss_huber = F.smooth_l1_loss(pred, future, beta=0.5)',
              explanation: 'گام ۱ (اتلاف نرم هوبر): مقاوم‌سازی گرادیان در برابر خودروهای با سرعت‌های متنوع و حذف ضربات ناگهانی به وزن‌ها.'
            },
            {
              line: 'diff2 = pred[:, 2:, :] - 2 * pred[:, 1:-1, :] + pred[:, :-2, :]',
              explanation: 'گام ۲ (قانون فیزیک و مشتق دوم): محاسبه بردار شتاب لحظه‌ای در تمام ۳۰ گام زمانی آینده خودرو: a_t = p_{t+1} - 2p_t + p_{t-1}.'
            },
            {
              line: 'loss_physics = torch.mean(torch.norm(diff2, p=2, dim=-1))',
              explanation: 'گام ۳ (جریمه شتاب‌های غیرواقعی): اجبار مدل به یادگیری مسیرهای پیوسته و روان بدون تکان‌ها یا پیچش‌های زاویه‌ای ناممکن.'
            },
            {
              line: 'loss = (loss_huber + 0.5*loss_ade + 0.25*loss_fde + 0.1*loss_physics) / 4',
              explanation: 'گام ۴: ترکیب ۴ گانه اهداف با تقسیم بر ۴ جهت تجمیع گرادیان در ۴ مینی‌بچ متوالی.'
            },
            {
              line: 'if (batch_idx + 1) % 4 == 0: scaler.step(optimizer); optimizer.zero_grad()',
              explanation: 'گام ۵ (اندازه دسته مؤثر ۱۲۸): اعمال هم‌زمان گرادیان ۱۲۸ خودرو که واریانس تصادفی را ۵۰٪ کاهش داده و نمودار را کاملاً نرم می‌کند.'
            }
          ],
          commonJuniorMistake: 'استفاده از L2 خالص که خطای نمونه‌های دورافتاده را به توان ۲ می‌رساند و موجب نوسانات شدید دندانه‌ای در نمودار می‌شود!'
        },
        {
          methodName: 'train_and_evaluate(self)',
          signature: 'def train_and_evaluate(self) -> FinalTrajectoryModel:',
          returnType: 'FinalTrajectoryModel: شیء مدل آموزش‌دیده با بارگذاری بهترین چک‌پوینت',
          location: 'core/final_trainer.py:148',
          difficulty: 'پیشرفته',
          stepGoal: 'حلقه اصلی آموزش با پایش پیوسته خطای اعتبارسنجی ADE و فعال‌سازی خودکار Early Stopping.',
          inputsExplained: [],
          lineByLineTeaching: [
            {
              line: 'improvement = (self.best_val_ade - mean_ade)',
              explanation: 'گام ۱: تفاضل بهترین خطای قبلی با خطای اپوک جاری.'
            },
            {
              line: 'is_best = improvement > self.cfg.early_stopping_min_delta',
              explanation: 'گام ۲: آیا خطا حداقل ۲ میلی‌متر (۰٫۰۰۲ متر) کمتر شده است؟ اگر بله، این اپوک واقعاً بهتر است، نه ناشی از نویز تصادفی.'
            },
            {
              line: 'if is_best: self.save_checkpoint(...); patience_counter = 0',
              explanation: 'گام ۳: اگر رکورد شکسته شد، چک‌پوینت ذخیره شده و شمارنده صبر صفر می‌شود.'
            },
            {
              line: 'else: patience_counter += 1',
              explanation: 'گام ۴: اگر پیشرفتی حاصل نشد، یک واحد به صبر اضافه می‌شود (Patience: x/8).'
            },
            {
              line: 'if patience_counter >= self.cfg.early_stopping_patience: break',
              explanation: 'گام ۵ (توقف هوشمند): به محض رسیدن به ۸ اپوک متوالی بدون پیشرفت، حلقه فورا قطع می‌شود.'
            },
            {
              line: 'ckpt = torch.load(self.cfg.save_path, weights_only=False); self.model.load_state_dict(...)',
              explanation: 'گام ۶ (بازیابی طلایی): وزن‌های ذخیره‌شده از آن اپوکی که کمترین خطا را ثبت کرده بود لود می‌شوند تا مدل خراب تحویل کاربر نشود!'
            }
          ],
          commonJuniorMistake: 'ذخیره کردن وزن‌های آخرین اپوک به جای بهترین اپوک! اگر مدل در اپوک ۳۶ متوقف شود ولی بهترین حالت در اپوک ۲۸ بوده باشد، نباید وزن‌های اپوک ۳۶ استفاده شوند.'
        },
        {
          methodName: 'validate(self, epoch: int)',
          signature: 'def validate(self, epoch: int) -> Dict[str, float]:',
          returnType: 'Dict[str, float]: {"val_ade": float, "val_fde": float, "miss_rate": float}',
          location: 'core/final_trainer.py:110',
          difficulty: 'متوسط',
          stepGoal: 'محاسبه خطای تعمیم مدل روی کل داده‌های اعتبارسنجی در محیط بدون گرادیان (torch.no_grad).',
          inputsExplained: [
            { param: 'epoch', type: 'int', desc: 'شماره اپوک جاری جهت چاپ لاگ' }
          ],
          lineByLineTeaching: [
            {
              line: 'self.model.eval()',
              explanation: 'غیرفعال‌سازی Dropout و قفل کردن لایه‌های BatchNorm/LayerNorm.'
            },
            {
              line: 'with torch.no_grad(): pred = self.model(past, neighbors, map_lanes)',
              explanation: 'محاسبه خروجی بدون ذخیره گراف محاسباتی در حافظه کارت گرافیک، که سرعت را ۵ برابر می‌کند.'
            },
            {
              line: 'mean_ade = ade(pred, future); mean_fde = fde(pred, future)',
              explanation: 'محاسبه فرمول اقلیدسی L2 در افق زمانی ۳۰ گام (۳ ثانیه).'
            }
          ],
          commonJuniorMistake: 'فراموش کردن self.model.train() پس از پایان ولیدیشن، که باعث می‌شود در اپوک بعدی یادگیری صورت نگیرد.'
        }
      ]
    },
    {
      classId: 'TrajectoryEvaluator',
      className: 'class TrajectoryEvaluator',
      fileLocation: 'python_src/evaluate/evaluator.py',
      inheritance: 'Object',
      roleBadge: 'ارزیابی آکادمیک و بنچمارک سخت‌افزار لبه (Stage 3)',
      classPurpose: 'موتور تخصصی ارزیابی مقایسه‌ای با مدل‌های مرجع جهان (Trajectron++, LaneGCN, TrajectoryNAS) و پروفایلر دقیق تأخیر سخت‌افزاری بر حسب میلی‌ثانیه و فریم بر ثانیه.',
      constructorExplained: {
        signature: 'def __init__(self, checkpoint_path: str = "checkpoints/best_trajectory_lnas.pt", dataset_name: str = "nuscenes", best_ops: Optional[Tuple[int, int, int]] = None):',
        description: 'لود مدل و بازیابی متادیتا از چک‌پوینت و اتصال به پروفایلر سخت‌افزاری.',
        params: [
          { name: 'checkpoint_path', desc: 'مسیر فایل وزن‌های ذخیره‌شده .pt' },
          { name: 'dataset_name', desc: 'نام دیتاست هدف (nuscenes یا argoverse)' }
        ]
      },
      methods: [
        {
          methodName: 'profile_hardware(self, num_runs: int = 100)',
          signature: 'def profile_hardware(self, num_runs: int = 100) -> Dict[str, Any]:',
          returnType: 'Dict[str, Any]: {"latency_ms_mean": float, "throughput_fps": float, "total_params_m": float}',
          location: 'evaluate/evaluator.py:165',
          difficulty: 'متوسط',
          stepGoal: 'اندازه‌گیری دقیق زمان استنتاج بلادرنگ روی برد یا کارت گرافیک با متدهای CUDA Event.',
          inputsExplained: [
            { param: 'num_runs', type: 'int', desc: 'تعداد تکرار برای محاسبه واریانس و صدک ۹۵ تاخیر' }
          ],
          lineByLineTeaching: [
            {
              line: 'for _ in range(15): _ = self.model(dummy_past, dummy_n, dummy_m)',
              explanation: 'گام ۱ (Warm-up): گرم کردن کش پردازنده گرافیکی قبل از شروع زمان‌سنجی تا اثر کلاک اولیه خنثی شود.'
            },
            {
              line: 'torch.cuda.synchronize(); t0 = time.perf_counter()',
              explanation: 'گام ۲: همگام‌سازی استریم CUDA برای دریافت زمان دقیق اجرای واقعی روی هسته‌ها.'
            },
            {
              line: 'mean_lat = np.mean(timings); p95_lat = np.percentile(timings, 95)',
              explanation: 'گام ۳: محاسبه میانگین و سخت‌گیرانه‌ترین زمان استنتاج برای ۵ درصد از سناریوهای بحرانی ترافیکی.'
            },
            {
              line: 'fps = 1000.0 / mean_lat',
              explanation: 'گام ۴: محاسبه نرخ فریم (تعداد سناریوهای قابل پردازش در هر ثانیه).'
            }
          ],
          commonJuniorMistake: 'استفاده از time.time() معمولی در پایتون بدون فراخوانی torch.cuda.synchronize()! چون CUDA ناهمگام (Async) کار می‌کنه، بدون همگام‌سازی زمان کذایی و غلط صفر صدم ثانیه می‌گیرید!'
        },
        {
          methodName: 'print_sota_comparison(...)',
          signature: 'def print_sota_comparison(eval_metrics, hw_metrics, search_runtime_seconds):',
          returnType: 'None: چاپ جدول استاندارد مقایسه مقالات ICRA و ECCV در کنسول',
          location: 'evaluate/evaluator.py:195',
          difficulty: 'مقدماتی',
          stepGoal: 'تطبیق و مقایسه برتری‌های مدل TrajectoryLNAS شما با مقالات پیشین در فرمت متنی رسمی.',
          inputsExplained: [
            { param: 'eval_metrics', type: 'Dict', desc: 'دیکشنری خطاهای محاسبه‌شده ADE, FDE, Miss Rate' },
            { param: 'hw_metrics', type: 'Dict', desc: 'دیکشنری آمار سخت‌افزار (تعداد پارامترها و FLOPs)' }
          ],
          lineByLineTeaching: [
            {
              line: 'print(f"TrajectoryLNAS [This Run] | {actual_ade} | {actual_fde} | {actual_params}M...")',
              explanation: 'چاپ ردیف اختصاصی بر اساس نتایج آزمایش جاری کاربر در مقایسه با جدول مقالات مرجع.'
            }
          ],
          commonJuniorMistake: 'چاپ اعداد ایستا (Hardcoded)؛ این متد در نسخه جدید ما به طور پویا به آزمایش شما متصل است.'
        }
      ]
    },
    {
      classId: 'SearchSuperNet',
      className: 'class SearchSuperNet(nn.Module)',
      fileLocation: 'python_src/models/search_supernet.py',
      inheritance: 'nn.Module',
      roleBadge: 'شبکه فراگیر و اشتراک وزن‌ها (Super-Net Architecture)',
      classPurpose: 'دربرگیرنده کل فضای جستجو (Search Space) شامل ۱۱ عملگر کاندید در ۳ شاخه زمانی، تعاملی و نقشه که به صورت موازی با متد MixedOp با هم رقابت می‌کنند.',
      constructorExplained: {
        signature: 'def __init__(self, model_config: ModelConfig):',
        description: 'ساخت انکودر ورودی، ماژول‌های کاندید زمانی، گراف‌های تعاملات، انکودر خطوط نقشه و دیکودر خروجی.',
        params: [
          { name: 'model_config', desc: 'پیکربندی شامل ابعاد ویژگی‌ها (d_model=128) و طول افق گذشته و آینده' }
        ]
      },
      methods: [
        {
          methodName: 'forward(self, past, neighbors, map_lanes, temperature=1.0)',
          signature: 'def forward(self, past, neighbors, map_lanes, temperature=1.0) -> torch.Tensor:',
          returnType: 'torch.Tensor: تانسور پیش‌بینی با ابعاد [Batch, 30, 2]',
          location: 'models/search_supernet.py:80',
          difficulty: 'پیشرفته',
          stepGoal: 'گذر داده‌ها از ترکیب محدب عملگرها با توزیع Softmax(α / τ).',
          inputsExplained: [
            { param: 'past', type: 'Tensor [B, 20, 2]', desc: 'تاریخچه حرکت خودرو در ۲ ثانیه گذشته' },
            { param: 'neighbors', type: 'Tensor [B, N, 20, 2]', desc: 'موقعیت خودروهای مجاور' },
            { param: 'map_lanes', type: 'Tensor [B, L, 10, 2]', desc: 'پلی‌لاین‌های خطوط نقشه معابر HD' },
            { param: 'temperature', type: 'float', desc: 'دمای سافت‌مکس برای کنترل میزان برندگی عملگرها' }
          ],
          lineByLineTeaching: [
            {
              line: 'a_temp, a_inter, a_map = self.arch_params.softmax_all(temperature)',
              explanation: 'اعمال سافت‌مکس با ضریب دمای جاری روی ضرایب کاندیدها.'
            },
            {
              line: 'feat_t = self.temporal_mixed_op(x, a_temp)',
              explanation: 'مجموع وزنی ۵ عملگر کاندید زمانی (Identity, GRU, LSTM, TCN, Attn) با وزن‌های آلفا.'
            },
            {
              line: 'feat_i = self.interaction_mixed_op(feat_t, neighbors, a_inter)',
              explanation: 'محاسبه هم‌زمان تاثیر خودروهای همسایه با ۵ عملگر گراف و استخری اجتماعی.'
            },
            {
              line: 'pred_traj = self.decoder(feat_fused)',
              explanation: 'نگاشت بردار ویژگی ادغام‌شده به ۳۰ جفت مختصات (X, Y) آینده.'
            }
          ],
          commonJuniorMistake: 'ضرب نکردن ضرایب آلفا در خروجی عملگرها؛ در سوپرنت خروجی نهایی باید ترکیب خطی خروجی تمام کاندیدها با وزن‌های آلفا باشد.'
        }
      ]
    }
  ];

  const currentClass = curriculum.find((c) => c.classId === selectedClassId) || curriculum[0];
  const currentMethod = currentClass.methods[selectedMethodIdx] || currentClass.methods[0];

  const filteredCurriculum = curriculum.filter((c) =>
    c.className.toLowerCase().includes(searchQuery.toLowerCase()) ||
    c.classPurpose.toLowerCase().includes(searchQuery.toLowerCase()) ||
    c.methods.some((m) => m.methodName.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Universal Page Header */}
      <div className="pb-4 border-b border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs text-cyan-400 font-mono mb-1">
            <span>OBJECT-ORIENTED ARCHITECTURE CURRICULUM</span>
            <span>·</span>
            <span>STEP-BY-STEP TEACHING</span>
            <span>·</span>
            <span>JUNIOR TO SENIOR</span>
          </div>
          <h1 className="text-xl font-bold text-white tracking-tight flex items-center gap-2.5">
            <GraduationCap className="w-5 h-5 text-cyan-400" />
            <span>آموزش تدریس گام‌به‌گام کلاس‌ها و توابع سورس‌کد (OOP Codebase Walkthrough)</span>
          </h1>
          <p className="text-sm text-slate-400 mt-1 max-w-3xl">
            مستندات تعاملی کلاس‌محور برای آموزش گام‌به‌گام لاجیک، پارامترهای ورودی/خروجی، جریان تانسورها و خطاهای رایج برنامه‌نویسان.
          </p>
        </div>

        {/* Search bar */}
        <div className="relative min-w-[260px]">
          <Search className="w-4 h-4 text-slate-400 absolute right-3 top-2.5" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="جستجوی کلاس یا متد (مثلاً train_epoch)..."
            className="w-full bg-slate-900 border border-slate-800 rounded-lg pr-9 pl-3 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
          />
        </div>
      </div>

      {/* Main Learning Workspace: Left (Class selector) / Right (Method step-by-step lecture) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Classes Directory (4 cols) */}
        <div className="lg:col-span-4 space-y-3">
          <div className="text-xs font-semibold text-slate-400 px-1 flex items-center justify-between">
            <span>کلاس‌های اصلی پروژه (Core Classes)</span>
            <span className="text-[10px] font-mono text-cyan-400">{curriculum.length} کلاس ماژولار</span>
          </div>

          <div className="space-y-2">
            {filteredCurriculum.map((cls) => {
              const isSelected = cls.classId === currentClass.classId;
              return (
                <div
                  key={cls.classId}
                  onClick={() => {
                    setSelectedClassId(cls.classId);
                    setSelectedMethodIdx(0);
                  }}
                  className={`p-3.5 rounded-xl border cursor-pointer transition-all ${
                    isSelected
                      ? 'bg-slate-900 border-cyan-500/50 shadow-lg shadow-cyan-950/20'
                      : 'bg-slate-950 border-slate-800/80 hover:border-slate-700 hover:bg-slate-900/50'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-mono text-xs font-bold text-white tracking-wide">{cls.className.split('(')[0]}</span>
                    <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/50 px-2 py-0.5 rounded border border-cyan-800/40">
                      {cls.methods.length} متد تدریس‌شده
                    </span>
                  </div>

                  <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">
                    {cls.classPurpose}
                  </p>

                  <div className="mt-2.5 pt-2 border-t border-slate-800/60 flex items-center justify-between text-[10px] font-mono text-slate-500">
                    <span>{cls.fileLocation}</span>
                    <span className="text-emerald-400">{cls.roleBadge.split(' ')[0]}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Column: Detailed Class Lecture and Method Walkthrough (8 cols) */}
        <div className="lg:col-span-8 space-y-5">
          {/* Class Overview Card */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-xl space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
              <div>
                <span className="text-[11px] font-mono text-cyan-400 uppercase tracking-wider block mb-1">
                  {currentClass.roleBadge}
                </span>
                <h2 className="text-base font-bold text-white font-mono">{currentClass.className}</h2>
                <span className="text-xs text-slate-400 font-mono">مسیر فایل: {currentClass.fileLocation}</span>
              </div>
              <span className="text-xs font-mono text-slate-300 bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 self-start">
                وراثت: {currentClass.inheritance}
              </span>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed bg-slate-950/60 p-3.5 rounded-lg border border-slate-800/70">
              <span className="font-semibold text-cyan-400 block mb-1">نقش کلاس در معماری TrajectoryLNAS:</span>
              {currentClass.classPurpose}
            </p>

            {/* Constructor details */}
            <div className="space-y-2">
              <span className="text-xs font-bold text-white flex items-center gap-2">
                <Code2 className="w-3.5 h-3.5 text-cyan-400" />
                <span>سازنده کلاس (Constructor __init__):</span>
              </span>
              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 font-mono text-xs text-emerald-300 overflow-x-auto flex items-center justify-between">
                <span>{currentClass.constructorExplained.signature}</span>
                <button
                  onClick={() => handleCopy(currentClass.constructorExplained.signature, 'ctor-sig')}
                  className="text-slate-400 hover:text-white mr-2 shrink-0"
                >
                  {copiedKey === 'ctor-sig' ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                </button>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px]">
                {currentClass.constructorExplained.params.map((p, idx) => (
                  <div key={idx} className="p-2.5 bg-slate-950/50 rounded border border-slate-800/70">
                    <span className="font-mono text-cyan-300 font-semibold block">{p.name}</span>
                    <span className="text-slate-400 mt-0.5 block">{p.desc}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Methods Selector Tabs */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-xl space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Flame className="w-4 h-4 text-amber-400" />
                <span>توابع و متدهای پیاده‌سازی‌شده (Step-by-Step Function Lessons)</span>
              </h3>
              <span className="text-xs font-mono text-slate-400">یک متد را برای تدریس خط به خط انتخاب کنید</span>
            </div>

            <div className="flex items-center gap-2 overflow-x-auto pb-1">
              {currentClass.methods.map((m, idx) => {
                const isSelected = idx === selectedMethodIdx;
                return (
                  <button
                    key={idx}
                    onClick={() => setSelectedMethodIdx(idx)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium whitespace-nowrap transition-all ${
                      isSelected
                        ? 'bg-cyan-500 text-slate-950 font-bold shadow-md'
                        : 'bg-slate-950 text-slate-400 hover:text-white border border-slate-800'
                    }`}
                  >
                    {m.methodName.split('(')[0]}()
                  </button>
                );
              })}
            </div>

            {/* Selected Method Detailed Lecture */}
            <div className="space-y-4 pt-2">
              <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-3">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs font-bold text-cyan-300">{currentMethod.methodName}</span>
                    <span className="text-[10px] font-mono text-amber-300 bg-amber-950/40 px-2 py-0.5 rounded border border-amber-800/40">
                      سطح: {currentMethod.difficulty}
                    </span>
                  </div>
                  <span className="text-[11px] font-mono text-slate-400">{currentMethod.location}</span>
                </div>

                <div className="p-2.5 bg-slate-900/90 rounded border border-slate-800 font-mono text-[11px] text-emerald-300 flex items-center justify-between overflow-x-auto">
                  <span>{currentMethod.signature}</span>
                  <button
                    onClick={() => handleCopy(currentMethod.signature, 'm-sig')}
                    className="text-slate-400 hover:text-white mr-2 shrink-0"
                  >
                    {copiedKey === 'm-sig' ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  </button>
                </div>

                <div className="p-3 bg-cyan-950/20 border border-cyan-500/20 rounded-lg text-xs text-cyan-200 leading-relaxed">
                  <span className="font-semibold block text-cyan-400 mb-0.5">هدف این تابع در پروژه چیست؟</span>
                  {currentMethod.stepGoal}
                </div>
              </div>

              {/* Input parameters explained */}
              {currentMethod.inputsExplained.length > 0 && (
                <div className="space-y-2">
                  <span className="text-xs font-semibold text-slate-300 block">آرگومان‌های ورودی تابع:</span>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                    {currentMethod.inputsExplained.map((inp, idx) => (
                      <div key={idx} className="p-2.5 bg-slate-950 rounded-lg border border-slate-800">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-mono text-cyan-300 font-semibold">{inp.param}</span>
                          <span className="text-[10px] font-mono text-slate-500 bg-slate-900 px-1.5 py-0.5 rounded">{inp.type}</span>
                        </div>
                        <p className="text-[11px] text-slate-400">{inp.desc}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Line-by-Line Teaching Section */}
              <div className="space-y-2">
                <span className="text-xs font-bold text-white flex items-center gap-2">
                  <GraduationCap className="w-4 h-4 text-cyan-400" />
                  <span>تدریس خط‌به‌خط لاجیک متد (Line-by-Line Walkthrough):</span>
                </span>

                <div className="space-y-2.5">
                  {currentMethod.lineByLineTeaching.map((step, idx) => (
                    <div key={idx} className="p-3.5 bg-slate-950 rounded-xl border border-slate-800 space-y-1.5">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] font-mono text-slate-400 bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
                          بخش {idx + 1} از {currentMethod.lineByLineTeaching.length}
                        </span>
                      </div>
                      <div className="p-2 bg-slate-900/90 rounded font-mono text-xs text-amber-300 border border-slate-800 overflow-x-auto">
                        {step.line}
                      </div>
                      <p className="text-xs text-slate-300 leading-relaxed pt-1">
                        {step.explanation}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Common Junior Mistake Warning */}
              <div className="p-3.5 bg-rose-950/20 border border-rose-500/20 rounded-xl flex items-start gap-3 text-xs text-rose-300 leading-relaxed">
                <HelpCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold text-rose-400 block mb-0.5">دام‌های کدنویسی و اشتباه متداول برنامه‌نویس جونیور:</span>
                  <span>{currentMethod.commonJuniorMistake}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
