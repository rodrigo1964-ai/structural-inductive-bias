#!/bin/bash
# ============================================================
# run_all.sh — Run all Case Studies for the 15Paper
#
# Usage:
#   ./run_all.sh              # Tests + figures
#   ./run_all.sh --tests      # Tests only (fast, uses saved .npz)
#   ./run_all.sh --figures    # Figures only (fast if .npz exist)
#   ./run_all.sh --full       # Re-run all experiments from scratch (~2h)
#
# Author: Rodolfo H. Rodrigo — UNSJ — 2026
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

MODE="${1:-all}"

header() {
    echo ""
    echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════${NC}"
}

run_step() {
    local label="$1"
    local dir="$2"
    local script="$3"

    echo -e "\n${BOLD}▶ ${label}${NC}"
    cd "$SCRIPT_DIR/$dir"
    if python3 "$script"; then
        echo -e "${GREEN}  ✓ ${label} — OK${NC}"
    else
        echo -e "${RED}  ✗ ${label} — FAILED${NC}"
        FAILURES=$((FAILURES + 1))
    fi
}

FAILURES=0
T_START=$SECONDS

# ── Tests ────────────────────────────────────────────────────
if [ "$MODE" = "all" ] || [ "$MODE" = "--tests" ]; then
    header "TESTS (from saved results/*.npz)"

    run_step "CS1 tests (envelope absorption)"      CaseStudy_1  test_caso1.py
    run_step "CS2 tests (nonlinearity, β>1)"         CaseStudy_2  test_caso2.py
    run_step "CS3 tests (HAM residual)"              CaseStudy_3  test_caso3.py
    run_step "CS4 tests (counterexample S=1)"        CaseStudy_4  test_caso4.py
    run_step "CS5 tests (analytical verification)"   CaseStudy_5  test_caso5.py
fi

# ── Figures ──────────────────────────────────────────────────
if [ "$MODE" = "all" ] || [ "$MODE" = "--figures" ]; then
    header "FIGURES (regenerated from saved results/*.npz)"

    run_step "CS1 figures (fig_exp1_*)"              CaseStudy_1  generate_figures.py
    run_step "CS2 figures (fig_exp2_*)"              CaseStudy_2  generate_figures.py
    run_step "CS3 figures (fig_exp3_* + barron)"     CaseStudy_3  generate_figures.py
    run_step "CS4 figures (fig_exp4_trivial)"        CaseStudy_4  generate_figures.py
    run_step "CS5 figures (fig_exp5_analytical)"     CaseStudy_5  generate_figures.py
fi

# ── Full re-run ──────────────────────────────────────────────
if [ "$MODE" = "--full" ]; then
    header "FULL RE-RUN (all experiments from scratch — ~2 hours)"

    run_step "Experiment 1 (envelope, 12×20 seeds)"  CaseStudy_1  caso1_envelope.py
    run_step "Experiment 2 (nonlinearity, 12×20)"    CaseStudy_2  caso2_nonlinearity.py
    run_step "Experiment 3 (HAM residual, 7K×20)"    CaseStudy_3  caso3_ham_residual.py
    run_step "Experiment 4 (counterexample, 12×20)"  CaseStudy_4  caso4_counterexample.py
    run_step "Experiment 5 (analytical, 7×20)"       CaseStudy_5  caso5_analytical.py
fi

# ── Summary ──────────────────────────────────────────────────
ELAPSED=$(( SECONDS - T_START ))
MIN=$(( ELAPSED / 60 ))
SEC=$(( ELAPSED % 60 ))

header "SUMMARY"
echo ""
if [ $FAILURES -eq 0 ]; then
    echo -e "  ${GREEN}${BOLD}✓ All OK${NC} — ${MIN}m ${SEC}s"
else
    echo -e "  ${RED}${BOLD}✗ ${FAILURES} failures${NC} — ${MIN}m ${SEC}s"
fi
echo ""

exit $FAILURES
