"use client";

import { AnalyzeRequest } from "@/types/jobfit";

interface SidebarOptionsProps {
  options: Omit<AnalyzeRequest, "job_posting">;
  onChange: (updated: Partial<Omit<AnalyzeRequest, "job_posting">>) => void;
}

export default function SidebarOptions({
  options,
  onChange,
}: SidebarOptionsProps) {
  return (
    <aside className="w-64 shrink-0 bg-gray-50 border-r border-gray-200 p-5 flex flex-col gap-5">
      <div>
        <h2 className="text-sm font-semibold text-gray-700 mb-3">Workflow</h2>
        <ol className="text-xs text-gray-500 space-y-1 list-decimal list-inside">
          <li>Paste a job posting</li>
          <li>Parse requirements</li>
          <li>Match against profile</li>
          <li>Generate cover letter</li>
        </ol>
      </div>

      <hr className="border-gray-200" />

      {/* Parser */}
      <div className="space-y-2">
        <h2 className="text-sm font-semibold text-gray-700">Parser</h2>
        <Toggle
          label="Use LLM parser"
          checked={options.use_llm}
          onChange={(v) => onChange({ use_llm: v })}
        />
      </div>

      {/* Generation */}
      <div className="space-y-2">
        <h2 className="text-sm font-semibold text-gray-700">Generation</h2>
        <Toggle
          label="Use LLM generation"
          checked={options.use_llm_generation}
          onChange={(v) => onChange({ use_llm_generation: v })}
        />
      </div>
    

      {/* Model */}
      <div className="space-y-1">
        <label className="text-xs font-medium text-gray-600">
          OpenAI model
        </label>
        <input
          type="text"
          value={options.model}
          disabled={!options.use_llm && !options.use_llm_generation}
          onChange={(e) => onChange({ model: e.target.value })}
          className="w-full text-xs border border-gray-300 rounded px-2 py-1.5 disabled:bg-gray-100 disabled:text-gray-400 focus:outline-none focus:ring-1 focus:ring-blue-400"
        />
      </div>

      <hr className="border-gray-200" />

      {/* Profile */}
      <div className="space-y-2">
        <h2 className="text-sm font-semibold text-gray-700">Profile</h2>
        <Toggle
          label="Use sample profile"
          checked={options.use_sample_profile}
          onChange={(v) => onChange({ use_sample_profile: v })}
        />
        <Toggle
          label="Use vector retrieval"
          checked={options.use_vector_retrieval}
          onChange={(v) => onChange({ use_vector_retrieval: v })}
        />
      </div>

      {/* Embedding provider */}
      <div className="space-y-1">
        <label className="text-xs font-medium text-gray-600">
          Embedding provider
        </label>
        <select
          value={options.embedding_provider}
          disabled={!options.use_vector_retrieval}
          onChange={(e) => onChange({ embedding_provider: e.target.value })}
          className="w-full text-xs border border-gray-300 rounded px-2 py-1.5 disabled:bg-gray-100 disabled:text-gray-400 focus:outline-none focus:ring-1 focus:ring-blue-400"
        >
          <option value="local">local</option>
          <option value="openai">openai</option>
        </select>
      </div>

      <hr className="border-gray-200" />

      {/* Approval */}
      <div className="space-y-2">
        <h2 className="text-sm font-semibold text-gray-700">Resume</h2>
        <Toggle
          label="Approve resume suggestions"
          checked={options.resume_suggestions_approved}
          onChange={(v) => onChange({ resume_suggestions_approved: v })}
        />
      </div>
    </aside>
  );
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-center justify-between cursor-pointer gap-2">
      <span className="text-xs text-gray-600">{label}</span>
      <button
        type="button"
        onClick={() => onChange(!checked)}
        className={`relative inline-flex h-5 w-9 shrink-0 rounded-full transition-colors duration-200 focus:outline-none ${
          checked ? "bg-blue-500" : "bg-gray-300"
        }`}
      >
        <span
          className={`inline-block h-4 w-4 mt-0.5 rounded-full bg-white shadow transform transition-transform duration-200 ${
            checked ? "translate-x-4" : "translate-x-0.5"
          }`}
        />
      </button>
    </label>
  );
}