"use client";

import { useState, useEffect } from "react";
import { AnalyzeRequest, AnalyzeResult } from "@/types/jobfit";
import { analyzeJobPosting, fetchSamplePosting } from "@/lib/api";
import SidebarOptions from "@/components/SidebarOptions";
import {
  TabAnalysis,
  TabMatch,
  TabPlan,
  TabSuggestions,
  TabCoverLetter,
  TabTrace,
  TabReport,
} from "@/components/ResultTabs";

const TABS = [
  { key: "analysis", label: "Job Analysis" },
  { key: "match", label: "Profile Match" },
  { key: "plan", label: "Learning Plan" },
  { key: "suggestions", label: "Resume Suggestions" },
  { key: "cover_letter", label: "Cover Letter" },
  { key: "trace", label: "Agent Trace" },
  { key: "report", label: "Report" },
];

const DEFAULT_OPTIONS: Omit<AnalyzeRequest, "job_posting"> = {
  use_llm: false,
  use_vector_retrieval: true,
  use_llm_generation: false,
  embedding_provider: "local",
  model: "gpt-4o-mini",
  use_sample_profile: true,
  resume_suggestions_approved: false,
};

export default function Home() {
  const [jobPosting, setJobPosting] = useState("");
  const [options, setOptions] = useState(DEFAULT_OPTIONS);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [activeTab, setActiveTab] = useState("analysis");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSamplePosting()
      .then(setJobPosting)
      .catch(() => {});
  }, []);

  const handleAnalyze = async () => {
    if (!jobPosting.trim()) {
      setError("Paste a job posting first.");
      return;
    }
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const data = await analyzeJobPosting({ job_posting: jobPosting, ...options });
      setResult(data);
      setActiveTab("analysis");
    } catch (e: any) {
      setError(e?.response?.data?.detail || "Something went wrong. Is the FastAPI server running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-white text-gray-900 overflow-hidden">
      {/* sidebar */}
      <SidebarOptions
        options={options}
        onChange={(updated) => setOptions((prev) => ({ ...prev, ...updated }))}
      />

      {/* Main */}
      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="px-8 py-5 border-b border-gray-200">
          <h1 className="text-xl font-bold text-gray-900">JobFit Agent</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            Paste a job posting and turn it into structured data for resume matching.
          </p>
        </header>

        <div className="flex-1 overflow-y-auto px-8 py-6 space-y-5">
          {/* Job postin */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5">
              Job Posting
            </label>
            <textarea
              value={jobPosting}
              onChange={(e) => setJobPosting(e.target.value)}
              rows={10}
              placeholder="Paste the full job posting here."
              className="w-full text-sm border border-gray-300 rounded-lg px-3 py-2.5 resize-none focus:outline-none focus:ring-2 focus:ring-blue-400"
            />
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg px-4 py-3">
              {error}
            </div>
          )}

          {/* Analyze */}
          <button
            onClick={handleAnalyze}
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white text-sm font-semibold py-2.5 rounded-lg transition-colors"
          >
            {loading ? "Running JobFit workflow..." : "Analyze Posting"}
          </button>

          {/* Result */}
          {result && (
            <div className="space-y-4">
              <div className="flex gap-1 border-b border-gray-200 overflow-x-auto">
                {TABS.map((tab) => (
                  <button
                    key={tab.key}
                    onClick={() => setActiveTab(tab.key)}
                    className={`shrink-0 px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                      activeTab === tab.key
                        ? "border-blue-500 text-blue-600"
                        : "border-transparent text-gray-500 hover:text-gray-700"
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              <div className="pb-8">
                {activeTab === "analysis" && <TabAnalysis result={result} />}
                {activeTab === "match" && <TabMatch result={result} />}
                {activeTab === "plan" && <TabPlan result={result} />}
                {activeTab === "suggestions" && <TabSuggestions result={result} />}
                {activeTab === "cover_letter" && <TabCoverLetter result={result} />}
                {activeTab === "trace" && <TabTrace result={result} />}
                {activeTab === "report" && <TabReport result={result} />}
              </div>
            </div>
          )}

          {!result && !loading && (
            <div className="text-center py-16 text-sm text-gray-400">
              Click Analyze Posting to start.
            </div>
          )}
        </div>
      </main>
    </div>
  );
}