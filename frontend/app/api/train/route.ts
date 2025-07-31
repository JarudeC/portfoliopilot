// API endpoint for initiating training jobs
import { NextRequest, NextResponse } from "next/server";
import { TrainingLogService } from "@/lib/services/training-logs";
import { getAuthenticatedUser } from "@/lib/auth/server";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

// Store job parameters for completion tracking
export const jobParameters = new Map<string, any>();

// Track logged jobs to prevent duplicates
export const loggedJobs = new Set<string>();

export async function POST(req: NextRequest) {
  try {
    // Check user authentication
    const user = await getAuthenticatedUser();
    if (!user) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }

    const body = await req.json();

    const r = await fetch(`${BACKEND}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const data = await r.json();

    // Cache parameters for result logging
    if (data.job_id) {
      jobParameters.set(data.job_id, body);
    }

    // Training completion logged in GET /api/train/[id]

    return NextResponse.json(data, { status: r.status });
  } catch (error) {
    console.error('Training request failed:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
