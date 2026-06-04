"use client";

import { AnalyzeResult } from "@/types/jobfit";

{/* Job Analysis */}
export function TabAnalysis({ result }: { result: AnalyzeResult }) {
  const { parsed_job } = result;
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-4">
        <MetricCard label="Role" value={parsed_job.role || "Unknown"} />
        <MetricCard label="Seniority" value={parsed_job.seniority || "Unknown"} />
        <MetricCard
          label="Skills Found"
          value={String(
            parsed_job.required_skills.length + parsed_job.preferred_skills.length
          )}
        />
      </div>

      <div className="grid grid-cols-2 gap-6">
        <Section title="Required Skills" items={parsed_job.required_skills} />
        <Section title="Preferred Skills" items={parsed_job.preferred_skills} />
        <Section title="Responsibilities" items={parsed_job.responsibilities} />
        <Section title="Keywords" items={parsed_job.keywords} />
      </div>
    </div>
  );
}

{/* Profile Match */}
export function TabMatch({ result }: { result: AnalyzeResult }) {
  const { matches, match_summary, gaps } = result;
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-4 gap-4">
        <MetricCard label="Avg Score" value={String(match_summary.average_score)} />
        <MetricCard label="Matched" value={String(match_summary.matched_count)} />
        <MetricCard label="Gaps" value={String(match_summary.gap_count)} />
        <MetricCard label="Total" value={String(match_summary.total_count)} />
      </div>

      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-xs text-gray-500 uppercase">
            <tr>
              {["Requirement", "Type", "Score", "Gap", "Source", "Method"].map((h) => (
                <th key={h} className="px-4 py-2 text-left">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {matches.map((m, i) => (
              <tr key={i} className="hover:bg-gray-50">
                <td className="px-4 py-2 font-medium">{m.requirement}</td>
                <td className="px-4 py-2 text-gray-500">{m.requirement_type}</td>
                <td className="px-4 py-2">
                  <ScoreBadge score={m.score} />
                </td>
                <td className="px-4 py-2">
                  {m.gap ? (
                    <span className="text-red-500 text-xs font-medium">Gap</span>
                  ) : (
                    <span className="text-green-500 text-xs font-medium">Match</span>
                  )}
                </td>
                <td className="px-4 py-2 text-gray-500 text-xs">{m.source || "-"}</td>
                <td className="px-4 py-2 text-gray-400 text-xs">{m.retrieval_method}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {gaps.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Gap Analysis</h3>
          <ul className="space-y-1">
            {gaps.map((gap, i) => (
              <li key={i} className="text-sm text-red-600 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-red-400 shrink-0" />
                {gap}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

{/* Learning Plan */}
export function TabPlan({ result }: { result: AnalyzeResult }) {
  const { learning_plan } = result;
  if (!learning_plan.length)
    return <Empty message="No learning plan was generated for this run." />;
  return (
    <ul className="space-y-2">
      {learning_plan.map((item, i) => (
        <li key={i} className="flex gap-3 text-sm text-gray-700">
          <span className="mt-0.5 text-blue-400 font-bold shrink-0">{i + 1}.</span>
          {item}
        </li>
      ))}
    </ul>
  );
}

{/* Resume Suggestion */}
export function TabSuggestions({ result }: { result: AnalyzeResult }) {
  const { resume_suggestions } = result;
  if (!resume_suggestions.length)
    return <Empty message="No resume suggestions were generated for this run." />;
  return (
    <div className="space-y-4">
      <p className="text-xs text-gray-500">
        Suggestions are only marked approved when the sidebar approval checkbox is
        enabled before analysis.
      </p>
      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-xs text-gray-500 uppercase">
            <tr>
              {["Requirement", "Current Evidence", "Suggested Bullet", "Approved"].map(
                (h) => (
                  <th key={h} className="px-4 py-2 text-left">{h}</th>
                )
              )}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {resume_suggestions.map((s, i) => (
              <tr key={i} className="hover:bg-gray-50 align-top">
                <td className="px-4 py-2 font-medium">{s.requirement}</td>
                <td className="px-4 py-2 text-gray-500 text-xs max-w-xs">{s.current_evidence}</td>
                <td className="px-4 py-2 text-gray-700 text-xs max-w-xs">{s.suggested_bullet}</td>
                <td className="px-4 py-2">
                  {s.approved ? (
                    <span className="text-green-500 text-xs font-medium">✓ Approved</span>
                  ) : (
                    <span className="text-gray-400 text-xs">Pending</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

{/* Cover Letter */}
export function TabCoverLetter({ result }: { result: AnalyzeResult }) {
  const { cover_letter, critic_score, revision_count } = result;
  if (!cover_letter) return <Empty message="No cover letter was generated for this run." />;
  return (
    <div className="space-y-4">
      {critic_score !== null && critic_score !== undefined && (
        <div className="flex items-center gap-4">
          <MetricCard label="Critic Score" value={`${critic_score}/100`} />
          <MetricCard label="Revisions" value={String(revision_count)} />
        </div>
      )}
      <div className="bg-gray-50 rounded-lg border border-gray-200 p-5">
        <pre className="text-sm text-gray-700 whitespace-pre-wrap font-sans leading-relaxed">
          {cover_letter}
        </pre>
      </div>
    </div>
  );
}

{/* Trace */}
export function TabTrace({ result }: { result: AnalyzeResult }) {
  const { trace } = result;
  if (!trace.length) return <Empty message="No trace available." />;
  return (
    <ol className="space-y-2">
      {trace.map((event, i) => (
        <li key={i} className="flex gap-3 text-sm">
          <span className="shrink-0 w-6 h-6 rounded-full bg-blue-100 text-blue-600 text-xs font-bold flex items-center justify-center">
            {i + 1}
          </span>
          <span className="text-gray-700 pt-0.5">{event}</span>
        </li>
      ))}
    </ol>
  );
}

{/* Report */}
export function TabReport({ result }: { result: AnalyzeResult }) {
  const { final_report } = result;
  if (!final_report) return <Empty message="No report available." />;
  return (
    <div className="prose prose-sm max-w-none">
      <pre className="text-sm text-gray-700 whitespace-pre-wrap font-sans leading-relaxed bg-gray-50 rounded-lg border border-gray-200 p-5">
        {final_report}
      </pre>
    </div>
  );
}


function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 px-4 py-3">
      <p className="text-xs text-gray-500">{label}</p>
      <p className="text-lg font-semibold text-gray-800 mt-0.5">{value}</p>
    </div>
  );
}

function Section({ title, items }: { title: string; items: string[] }) {
  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-700 mb-2">{title}</h3>
      {items.length ? (
        <ul className="space-y-1">
          {items.map((item, i) => (
            <li key={i} className="text-sm text-gray-600 flex gap-2 items-start">
              <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-blue-400 shrink-0" />
              {item}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-gray-400">None found.</p>
      )}
    </div>
  );
}

function ScoreBadge({ score }: { score: number }) {
  const color =
    score >= 4
      ? "bg-green-100 text-green-700"
      : score >= 3
      ? "bg-yellow-100 text-yellow-700"
      : "bg-red-100 text-red-700";
  return (
    <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${color}`}>
      {score}/5
    </span>
  );
}

function Empty({ message }: { message: string }) {
  return (
    <div className="text-center py-12 text-sm text-gray-400">{message}</div>
  );
}