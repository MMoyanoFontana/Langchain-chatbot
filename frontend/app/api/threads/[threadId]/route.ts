import { NextRequest } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export async function GET(
  _request: Request,
  context: { params: Promise<{ threadId: string }> }
) {
  const { threadId } = await context.params;
  const backendEndpoint = new URL(
    `/users/dev/current/threads/${threadId}`,
    BACKEND_URL
  );

  try {
    const upstreamResponse = await fetch(backendEndpoint, {
      headers: { Accept: "application/json" },
      method: "GET",
      cache: "no-store",
    });

    if (!upstreamResponse.ok) {
      const errorText = await upstreamResponse.text();
      return Response.json(
        { error: errorText || "Backend thread request failed." },
        { status: upstreamResponse.status || 502 }
      );
    }

    const payload = await upstreamResponse.json();
    return Response.json(payload, {
      headers: { "Cache-Control": "no-store" },
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
    });
  } catch {
    return Response.json(
      { error: "Unable to reach threads backend." },
      { status: 502 }
    );
  }
}

type UpdateThreadBody = {
  title?: string;
};

export async function PATCH(
  request: NextRequest,
  context: { params: Promise<{ threadId: string }> }
) {
  const { threadId } = await context.params;
  let body: UpdateThreadBody;

  try {
    body = (await request.json()) as UpdateThreadBody;
  } catch {
    return Response.json({ error: "Invalid request body." }, { status: 400 });
  }

  const title = body.title?.trim() ?? "";
  if (!title) {
    return Response.json({ error: "Title is required." }, { status: 400 });
  }

  const backendEndpoint = new URL(
    `/users/dev/current/threads/${threadId}`,
    BACKEND_URL
  );

  try {
    const upstreamResponse = await fetch(backendEndpoint, {
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      method: "PATCH",
      body: JSON.stringify({ title }),
      cache: "no-store",
    });

    if (!upstreamResponse.ok) {
      if (upstreamResponse.status === 405) {
        return Response.json(
          {
            error:
              "Thread actions are unavailable on the current backend process. Restart backend to load PATCH/DELETE thread routes.",
          },
          { status: 409 }
        );
      }
      const errorText = await upstreamResponse.text();
      return Response.json(
        { error: errorText || "Backend thread update failed." },
        { status: upstreamResponse.status || 502 }
      );
    }

    const payload = await upstreamResponse.json();
    return Response.json(payload, {
      headers: { "Cache-Control": "no-store" },
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
    });
  } catch {
    return Response.json(
      { error: "Unable to reach threads backend." },
      { status: 502 }
    );
  }
}

export async function DELETE(
  _request: Request,
  context: { params: Promise<{ threadId: string }> }
) {
  const { threadId } = await context.params;
  const backendEndpoint = new URL(
    `/users/dev/current/threads/${threadId}`,
    BACKEND_URL
  );

  try {
    const upstreamResponse = await fetch(backendEndpoint, {
      headers: { Accept: "application/json" },
      method: "DELETE",
      cache: "no-store",
    });

    if (!upstreamResponse.ok) {
      if (upstreamResponse.status === 405) {
        return Response.json(
          {
            error:
              "Thread actions are unavailable on the current backend process. Restart backend to load PATCH/DELETE thread routes.",
          },
          { status: 409 }
        );
      }
      const errorText = await upstreamResponse.text();
      return Response.json(
        { error: errorText || "Backend thread delete failed." },
        { status: upstreamResponse.status || 502 }
      );
    }

    return new Response(null, { status: 204 });
  } catch {
    return Response.json(
      { error: "Unable to reach threads backend." },
      { status: 502 }
    );
  }
}
