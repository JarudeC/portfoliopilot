import { NextRequest, NextResponse } from "next/server";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function GET(
  _req: NextRequest,
  { params }: { params: { id: string } }
) {
  const { id } = await params;
  const r = await fetch(`${BACKEND}/train/${id}`);
  const data = await r.json();
  return NextResponse.json(data, { status: r.status });
}
