#!/bin/bash
# ─────────────────────────────────────────────────────────────
# NHL WAR – Update Player Cards
# Double-click this file in Finder to rebuild the app
# with the latest skaters.csv / goalies.csv data.
# ─────────────────────────────────────────────────────────────

# Move to the folder this script lives in
cd "$(dirname "$0")"

echo ""
echo "════════════════════════════════════"
echo "  NHL WAR  –  Data Update"
echo "════════════════════════════════════"
echo ""

# ── Check Python ──────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "❌  Python 3 is not installed on this Mac."
  echo "    Please install it from https://www.python.org and try again."
  echo ""
  read -p "Press Enter to close..."
  exit 1
fi

# ── Install dependencies if missing ──────────────────────────
echo "⏳  Checking dependencies..."
python3 -c "import pandas, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
  echo "📦  Installing pandas & numpy (one-time setup)..."
  pip3 install pandas numpy --break-system-packages -q
  if [ $? -ne 0 ]; then
    pip3 install pandas numpy -q
  fi
fi

echo "✅  Dependencies ready."
echo ""

# ── Check CSV files exist ─────────────────────────────────────
if [ ! -f "2025-2026/skaters.csv" ]; then
  echo "❌  skaters.csv not found in the 2025-2026 folder."
  echo "    Make sure skaters.csv is inside the 2025-2026 subfolder."
  echo ""
  read -p "Press Enter to close..."
  exit 1
fi

if [ ! -f "2025-2026/goalies.csv" ]; then
  echo "❌  goalies.csv not found in the 2025-2026 folder."
  echo "    Make sure goalies.csv is inside the 2025-2026 subfolder."
  echo ""
  read -p "Press Enter to close..."
  exit 1
fi

# ── Run the build ─────────────────────────────────────────────
echo "🏒  Building player cards from CSV data..."
echo ""
python3 build_war.py

if [ $? -eq 0 ]; then
  echo ""
  echo "════════════════════════════════════"
  echo "  ✅  Done! NHL_WAR_Cards.html"
  echo "      has been updated."
  echo "════════════════════════════════════"
  echo ""
  echo "  Refresh the file in your browser to see the new data."
  echo ""
  # Auto-open the HTML file
  open "NHL_WAR_Cards.html"
else
  echo ""
  echo "════════════════════════════════════"
  echo "  ❌  Something went wrong."
  echo "      See error above for details."
  echo "════════════════════════════════════"
  echo ""
fi

read -p "Press Enter to close..."
