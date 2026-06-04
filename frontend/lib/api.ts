import axios from "axios";
import { AnalyzeRequest, AnalyzeResult } from "@/types/jobfit";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function analyzeJobPosting(
  req: AnalyzeRequest
): Promise<AnalyzeResult> {
  const response = await axios.post<AnalyzeResult>(`${API_BASE}/analyze`, req);
  return response.data;
}

export async function fetchSamplePosting(): Promise<string> {
  const response = await axios.get<{ content: string }>(
    `${API_BASE}/sample/posting`
  );
  return response.data.content;
}