/**
 * API route for managing user API keys.
 *
 * Endpoints:
 * - GET: Check if user has an API key and get preview
 * - POST: Save or update an API key
 * - DELETE: Remove an API key
 *
 * All endpoints require authentication.
 * Keys are encrypted before storage and never returned in full.
 */

import { NextRequest, NextResponse } from 'next/server';
import { requireAuth, createAuthError } from '@/lib/auth/server';
import { ApiKeyService, ApiProvider } from '@/lib/services/api-keys';

// Supported providers - currently only Anthropic
const VALID_PROVIDERS: ApiProvider[] = ['anthropic'];

/**
 * GET /api/settings/api-key?provider=anthropic
 *
 * Check if user has an API key for the given provider.
 * Returns key info (preview, dates) without the actual key.
 */
export async function GET(req: NextRequest) {
  try {
    await requireAuth();

    const { searchParams } = new URL(req.url);
    const provider = searchParams.get('provider') as ApiProvider;

    // Validate provider parameter
    if (!provider || !VALID_PROVIDERS.includes(provider)) {
      return NextResponse.json(
        { error: `Invalid provider. Supported: ${VALID_PROVIDERS.join(', ')}` },
        { status: 400 }
      );
    }

    const keyService = new ApiKeyService(true);
    const keyInfo = await keyService.getKeyInfo(provider);

    if (!keyInfo) {
      return NextResponse.json({
        hasKey: false,
        provider,
      });
    }

    return NextResponse.json({
      hasKey: true,
      provider,
      keyPreview: keyInfo.key_preview,
      createdAt: keyInfo.created_at,
      updatedAt: keyInfo.updated_at,
    });
  } catch (error) {
    console.error('Failed to get API key info:', error);

    if (error instanceof Error && error.message.includes('Authentication')) {
      return createAuthError();
    }

    return NextResponse.json(
      { error: 'Failed to get API key information' },
      { status: 500 }
    );
  }
}

/**
 * POST /api/settings/api-key
 *
 * Save or update an API key.
 * Body: { provider: 'anthropic', apiKey: 'sk-ant-...' }
 */
export async function POST(req: NextRequest) {
  try {
    await requireAuth();

    const body = await req.json();
    const { provider, apiKey } = body as { provider: ApiProvider; apiKey: string };

    // Validate provider
    if (!provider || !VALID_PROVIDERS.includes(provider)) {
      return NextResponse.json(
        { error: `Invalid provider. Supported: ${VALID_PROVIDERS.join(', ')}` },
        { status: 400 }
      );
    }

    // Validate API key is provided
    if (!apiKey || typeof apiKey !== 'string' || apiKey.trim().length === 0) {
      return NextResponse.json(
        { error: 'API key is required' },
        { status: 400 }
      );
    }

    const keyService = new ApiKeyService(true);

    // Save the key (validation happens in service)
    await keyService.saveKey(provider, apiKey.trim());

    // Get the updated key info to return
    const keyInfo = await keyService.getKeyInfo(provider);

    return NextResponse.json({
      success: true,
      message: 'API key saved successfully',
      keyPreview: keyInfo?.key_preview,
    });
  } catch (error) {
    console.error('Failed to save API key:', error);

    if (error instanceof Error) {
      if (error.message.includes('Authentication')) {
        return createAuthError();
      }
      if (error.message.includes('Invalid API key format')) {
        return NextResponse.json(
          { error: error.message },
          { status: 400 }
        );
      }
    }

    return NextResponse.json(
      { error: 'Failed to save API key' },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/settings/api-key?provider=anthropic
 *
 * Remove an API key for the given provider.
 */
export async function DELETE(req: NextRequest) {
  try {
    await requireAuth();

    const { searchParams } = new URL(req.url);
    const provider = searchParams.get('provider') as ApiProvider;

    // Validate provider parameter
    if (!provider || !VALID_PROVIDERS.includes(provider)) {
      return NextResponse.json(
        { error: `Invalid provider. Supported: ${VALID_PROVIDERS.join(', ')}` },
        { status: 400 }
      );
    }

    const keyService = new ApiKeyService(true);
    await keyService.deleteKey(provider);

    return NextResponse.json({
      success: true,
      message: 'API key deleted successfully',
    });
  } catch (error) {
    console.error('Failed to delete API key:', error);

    if (error instanceof Error && error.message.includes('Authentication')) {
      return createAuthError();
    }

    return NextResponse.json(
      { error: 'Failed to delete API key' },
      { status: 500 }
    );
  }
}
