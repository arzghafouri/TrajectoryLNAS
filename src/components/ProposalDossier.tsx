import React, { useState } from 'react';
import { FileText, User, GraduationCap, CheckCircle, Bookmark, DollarSign, BookOpen, Layers } from 'lucide-react';
import { THESIS_DATA, THESIS_SECTIONS } from '../data/thesisDossier';

export const ProposalDossier: React.FC = () => {
  const [activeSectionId, setActiveSectionId] = useState<string>('4-1');

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Official Worksheet Title Banner */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 shadow-xl relative overflow-hidden">
        {/* Subtle watermark badge */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-cyan-400 mb-1">
              <span>{THESIS_DATA.docNumber}</span>
              <span>·</span>
              <span>شماره ثبت ایرانداک: {THESIS_DATA.irandocNumber}</span>
              <span>·</span>
              <span>شماره ثبت پژوهش: {THESIS_DATA.researchNumber}</span>
              <span>·</span>
              <span>تاریخ ثبت: {THESIS_DATA.proposalDate}</span>
            </div>
            <h1 className="text-xl md:text-2xl font-bold text-white tracking-tight">
              کاربرگ پیشنهاد پایان‌نامه کارشناسی ارشد
            </h1>
            <p className="text-sm font-medium text-cyan-300 mt-1">
              {THESIS_DATA.titleFa}
            </p>
            <p className="text-xs font-mono text-slate-400 mt-0.5">
              {THESIS_DATA.titleEn}
            </p>
          </div>

          <div className="flex flex-col items-start md:items-end gap-1.5 text-xs text-slate-300 font-mono">
            <span className="px-2.5 py-1 rounded bg-slate-800 border border-slate-700/80">
              نوع پایان‌نامه: <strong className="text-cyan-400">{THESIS_DATA.thesisType}</strong>
            </span>
            <span className="text-slate-400">
              مدت اجرا: {THESIS_DATA.duration} | تعداد واحد: {THESIS_DATA.credits}
            </span>
          </div>
        </div>

        {/* Student & Advisors Info Matrix */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4 pt-2 text-xs">
          {/* Student */}
          <div className="p-3.5 bg-slate-950/70 border border-slate-800 rounded-lg space-y-1">
            <div className="flex items-center gap-1.5 text-cyan-400 font-semibold mb-1">
              <User className="w-3.5 h-3.5" />
              <span>مشخصات دانشجو</span>
            </div>
            <div className="text-slate-200 font-bold text-sm">{THESIS_DATA.student.name}</div>
            <div className="text-slate-400">شماره دانشجویی: {THESIS_DATA.student.studentId}</div>
            <div className="text-slate-400">{THESIS_DATA.student.major}</div>
            <div className="text-slate-500 font-mono text-[11px] pt-1">
              {THESIS_DATA.student.email}
            </div>
          </div>

          {/* Supervisor */}
          <div className="p-3.5 bg-slate-950/70 border border-slate-800 rounded-lg space-y-1">
            <div className="flex items-center gap-1.5 text-emerald-400 font-semibold mb-1">
              <GraduationCap className="w-3.5 h-3.5" />
              <span>{THESIS_DATA.supervisors[0].role}</span>
            </div>
            <div className="text-slate-200 font-bold text-sm">
              {THESIS_DATA.supervisors[0].name}
            </div>
            <div className="text-slate-400">{THESIS_DATA.supervisors[0].university}</div>
            <div className="text-slate-400">{THESIS_DATA.supervisors[0].field}</div>
            <div className="text-slate-500 font-mono text-[11px] pt-1">
              {THESIS_DATA.supervisors[0].email}
            </div>
          </div>

          {/* Advisor */}
          <div className="p-3.5 bg-slate-950/70 border border-slate-800 rounded-lg space-y-1">
            <div className="flex items-center gap-1.5 text-amber-400 font-semibold mb-1">
              <GraduationCap className="w-3.5 h-3.5" />
              <span>{THESIS_DATA.supervisors[1].role}</span>
            </div>
            <div className="text-slate-200 font-bold text-sm">
              {THESIS_DATA.supervisors[1].name}
            </div>
            <div className="text-slate-400">{THESIS_DATA.supervisors[1].university}</div>
            <div className="text-slate-400">{THESIS_DATA.supervisors[1].field}</div>
            <div className="text-slate-500 font-mono text-[11px] pt-1">
              {THESIS_DATA.supervisors[1].email}
            </div>
          </div>
        </div>

        {/* Keywords */}
        <div className="mt-4 pt-3 border-t border-slate-800/80 text-xs">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-slate-400 font-semibold">واژه‌های کلیدی:</span>
            {THESIS_DATA.keywordsFa.map((kw, idx) => (
              <span key={idx} className="text-slate-300">
                {kw} {idx < THESIS_DATA.keywordsFa.length - 1 && '·'}
              </span>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-2 font-mono text-[11px] text-slate-400 mt-1">
            <span className="text-slate-500">Keywords:</span>
            {THESIS_DATA.keywordsEn.map((kw, idx) => (
              <span key={idx} className="text-slate-400">
                {kw} {idx < THESIS_DATA.keywordsEn.length - 1 && '·'}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Main Sections Accordion / Explorer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Navigation Sidebar (4 cols) */}
        <div className="lg:col-span-4 bg-slate-900/90 border border-slate-800 rounded-xl p-3 space-y-1.5 shadow-xl h-fit">
          <span className="text-[11px] font-mono text-slate-400 px-3 py-1.5 block">
            بخش‌های پیشنهاده (Sections)
          </span>

          {THESIS_SECTIONS.map((sec) => (
            <button
              key={sec.id}
              onClick={() => setActiveSectionId(sec.id)}
              className={`w-full text-right px-3 py-2 text-xs font-medium rounded-lg transition-colors flex items-center justify-between ${
                activeSectionId === sec.id
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              <span className="truncate">{sec.title}</span>
              <BookOpen className="w-3.5 h-3.5 shrink-0 opacity-60" />
            </button>
          ))}
        </div>

        {/* Section Content Reader (8 cols) */}
        <div className="lg:col-span-8 bg-slate-900/90 border border-slate-800 rounded-xl p-6 space-y-4 shadow-xl">
          {(() => {
            const currentSec =
              THESIS_SECTIONS.find((s) => s.id === activeSectionId) || THESIS_SECTIONS[0];
            return (
              <div>
                <h2 className="text-base font-bold text-white pb-3 border-b border-slate-800">
                  {currentSec.title}
                </h2>
                <div className="mt-4 text-xs md:text-sm text-slate-300 leading-relaxed whitespace-pre-line">
                  {currentSec.contentFa}
                </div>
              </div>
            );
          })()}
        </div>
      </div>

      {/* Approved Thesis Budget Section (بخش ۵ کاربرگ شماره ۲) */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 space-y-4 shadow-xl">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2 text-white font-bold text-base">
            <DollarSign className="w-4 h-4 text-emerald-400" />
            <span>بخش ۵: جدول هزینه‌های پایان‌نامه (Approved Thesis Budget)</span>
          </div>
          <span className="text-xs font-mono text-emerald-400 font-bold">
            جمع کل: {THESIS_DATA.totalBudgetRials}
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-right text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 text-[11px]">
                <th className="py-2.5 px-3">ردیف</th>
                <th className="py-2.5 px-3">نوع هزینه</th>
                <th className="py-2.5 px-3">عنوان و شرح</th>
                <th className="py-2.5 px-3">مبلغ مصوب (ریال)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {THESIS_DATA.budget.map((b) => (
                <tr key={b.row} className="hover:bg-slate-800/30 text-slate-300">
                  <td className="py-3 px-3">{b.row}</td>
                  <td className="py-3 px-3 text-slate-400">{b.type}</td>
                  <td className="py-3 px-3 text-slate-200">{b.title}</td>
                  <td className="py-3 px-3 text-emerald-400 tabular-nums">{b.costRials}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between text-xs text-slate-400">
          <div className="flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-cyan-400" />
            <span>تأیید اولیه پایان‌نامه مورخ ۱۴۰۴/۱۲/۰۵ به امضای دانشجو و اساتید راهنما و مشاور رسیده است.</span>
          </div>
        </div>
      </div>
    </div>
  );
};
