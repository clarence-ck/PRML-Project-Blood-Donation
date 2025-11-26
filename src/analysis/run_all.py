"""Master Orchestrator - Execute all POST-TRAINING analysis modules.

PREREQUISITE: Must have already completed:
1. python Dataset.py (generate raw data)
2. python -m src.analysis.eda (analyze raw data)
3. python process_data.py (engineer features)
4. python -m src.pipeline.run (train models)

This script runs AFTER training to generate evaluation reports:
1. Model Evaluation
2. Interpretability Analysis  
3. Subgroup Analysis
4. Report Generation
5. Success Criteria Validation
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

print("=" * 70)
print(" " * 15 + "BLOOD DONOR PREDICTION ANALYSIS")
print("=" * 70)


def run_evaluation():
    """Step 1: Model Evaluation."""
    print("\n" + "=" * 70)
    print("[STEP 1/5] MODEL EVALUATION")
    print("=" * 70)
    
    try:
        from src.analysis.evaluation import run_evaluation_analysis
        run_evaluation_analysis()
        print("\n✅ Evaluation completed successfully")
        return True
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        traceback.print_exc()
        return False


def run_interpretability():
    """Step 2: Interpretability Analysis."""
    print("\n" + "=" * 70)
    print("[STEP 2/5] INTERPRETABILITY ANALYSIS")
    print("=" * 70)
    
    try:
        # Check if interpretability module exists from old structure
        if Path("src/analysis/interpretability.py").exists():
            from src.analysis.interpretability import run_interpretability_analysis
            run_interpretability_analysis()
            print("\n✅ Interpretability analysis completed successfully")
            return True
        else:
            print("\n⚠️  Interpretability module not found, skipping...")
            return True
    except Exception as e:
        print(f"\n❌ Interpretability analysis failed: {e}")
        traceback.print_exc()
        return False


def run_subgroups():
    """Step 3: Subgroup Fairness Analysis."""
    print("\n" + "=" * 70)
    print("[STEP 3/5] SUBGROUP FAIRNESS ANALYSIS")
    print("=" * 70)
    
    try:
        # Check if subgroups module exists from old structure
        if Path("src/analysis/subgroups.py").exists():
            from src.analysis.subgroups import run_subgroup_analysis
            run_subgroup_analysis()
            print("\n✅ Subgroup analysis completed successfully")
            return True
        else:
            print("\n⚠️  Subgroup analysis module not found, skipping...")
            return True
    except Exception as e:
        print(f"\n❌ Subgroup analysis failed: {e}")
        traceback.print_exc()
        return False


def run_report_generation():
    """Step 4: Generate Final Report."""
    print("\n" + "=" * 70)
    print("[STEP 4/5] FINAL REPORT GENERATION")
    print("=" * 70)
    
    try:
        from src.analysis.report_generator import generate_final_report
        generate_final_report()
        print("\n✅ Report generation completed successfully")
        return True
    except Exception as e:
        print(f"\n❌ Report generation failed: {e}")
        traceback.print_exc()
        return False


def run_validation():
    """Step 5: Validate Success Criteria."""
    print("\n" + "=" * 70)
    print("[STEP 5/5] SUCCESS CRITERIA VALIDATION")
    print("=" * 70)
    
    try:
        from src.analysis.validate_criteria import validate_all_criteria
        result = validate_all_criteria()
        print("\n✅ Validation completed successfully")
        return result.get("all_passed", False)
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Execute all analysis steps in sequence."""
    steps = [
        ("Model Evaluation", run_evaluation),
        ("Interpretability", run_interpretability),
        ("Subgroup Analysis", run_subgroups),
        ("Report Generation", run_report_generation),
        ("Success Criteria Validation", run_validation),
    ]
    
    results = {}
    
    for step_name, step_func in steps:
        try:
            success = step_func()
            results[step_name] = success
        except KeyboardInterrupt:
            print("\n\n⚠️  Pipeline interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unexpected error in {step_name}: {e}")
            traceback.print_exc()
            results[step_name] = False
    
    # Final summary
    print("\n" + "=" * 70)
    print(" " * 25 + "EXECUTION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nCompleted Steps: {passed}/{total}\n")
    
    for step_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}  {step_name}")
    
    print("\n" + "=" * 70)
    
    if passed == total:
        print("\n🎉 ALL ANALYSES COMPLETED SUCCESSFULLY! 🎉\n")
        print("Next steps:")
        print("  1. Review the final report: reports/FINAL_REPORT.md")
        print("  2. Examine visualizations in: reports/*/figures/")
        print("  3. Check validation results: reports/success_criteria_validation.json")
        print("  4. Deploy best model: models/best_model.pkl")
    else:
        print("\n⚠️  SOME ANALYSES FAILED - Review error messages above\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
