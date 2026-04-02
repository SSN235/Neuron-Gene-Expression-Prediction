#!/bin/bash
# NGEP Validator - Automated Deployment Script
# This script deploys the entire validator to Cloudflare Workers + Pages

set -e

echo "=========================================="
echo "NGEP Validator - Cloudflare Deployment"
echo "=========================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Install from https://nodejs.org/"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ npm not found."
    exit 1
fi

echo "✓ Node.js: $(node --version)"
echo "✓ npm: $(npm --version)"
echo ""

# Check Wrangler
echo "Installing Wrangler CLI..."
npm install -g @cloudflare/wrangler > /dev/null 2>&1 || true

if ! command -v wrangler &> /dev/null; then
    echo "❌ Wrangler installation failed. Install manually:"
    echo "   npm install -g @cloudflare/wrangler"
    exit 1
fi

echo "✓ Wrangler: $(wrangler --version)"
echo ""

# Login to Cloudflare
echo "Logging in to Cloudflare..."
echo "A browser window will open. Log in or create a free account."
echo ""
wrangler login || {
    echo "❌ Cloudflare login failed"
    exit 1
}

echo ""
echo "=========================================="
echo "Deploying Backend (Cloudflare Workers)..."
echo "=========================================="
echo ""

# Deploy backend
wrangler deploy

echo ""
echo "✓ Backend deployed!"
echo ""

# Get backend URL
BACKEND_URL=$(wrangler deployments list 2>/dev/null | head -1 || echo "")
if [ -z "$BACKEND_URL" ]; then
    BACKEND_URL="https://ngep-validator.your-domain.workers.dev"
fi

echo "Backend URL: $BACKEND_URL"
echo ""

# Create frontend folder
mkdir -p frontend

# Copy index.html
if [ -f "index.html" ]; then
    cp index.html frontend/
    echo "✓ Frontend files copied"
else
    echo "❌ index.html not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "Deploying Frontend (Cloudflare Pages)..."
echo "=========================================="
echo ""

wrangler pages deploy frontend/

echo ""
echo "✓ Frontend deployed!"
echo ""

echo "=========================================="
echo "✅ DEPLOYMENT COMPLETE!"
echo "=========================================="
echo ""
echo "Your NGEP Validator is now LIVE!"
echo ""
echo "📝 NEXT STEPS:"
echo ""
echo "1. Update index.html with your backend URL"
echo "   Change: window.NGEP_BACKEND_URL = '...'"
echo "   To: window.NGEP_BACKEND_URL = '$BACKEND_URL'"
echo ""
echo "2. Redeploy frontend:"
echo "   wrangler pages deploy frontend/"
echo ""
echo "3. Open the Pages URL (from output above)"
echo ""
echo "4. Click 'Upload Model' and select your .pt/.pkl file"
echo ""
echo "5. Click 'Run Validation'"
echo ""
echo "6. Share the URL with others (no login needed!)"
echo ""
echo "=========================================="
