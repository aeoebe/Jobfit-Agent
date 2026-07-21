export interface RequirementMatch {
  requirement: string;
  requirement_type: string;
  matched_evidence: string;
  source: string;
  score: number;
  gap: boolean;
  retrieval_method: string;
}

export interface MatchSummary {
  average_score: number;
  matched_count: number;
  gap_count: number;
  total_count: number;
}

export interface ResumeSuggestion {
  requirement: string;
  current_evidence: string;
  suggested_bullet: string;
  approved: boolean;
}

export interface ParsedJob {
  role: string;
  seniority: string;
  required_skills: string[];
  preferred_skills: string[];
  responsibilities: string[];
  keywords: string[];
}

export interface AnalyzeResult {
  parsed_job: ParsedJob;
  matches: RequirementMatch[];
  match_summary: MatchSummary;
  gaps: string[];
  next_step: string;
  learning_plan: string[];
  resume_suggestions: ResumeSuggestion[];
  cover_letter: string;
  critic_score: number | null;
  revision_count: number;
  final_report: string;
  trace: string[];
}

export interface AnalyzeRequest {
  job_posting: string;
  use_llm: boolean;
  llm_provider: string;
  use_vector_retrieval: boolean;
  use_llm_generation: boolean;
  embedding_provider: string;
  model: string;
  use_sample_profile: boolean;
  resume_suggestions_approved: boolean;
}
