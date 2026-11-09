"""
Project SHIFT: Multi-Format Evaluation Report Generator.
Component 8 of SHIFT v0.3 Scientific Benchmark Infrastructure.

Generates reproducible, publication-grade benchmark evaluation reports:
1. Markdown reports with formatted comparison tables for documentation and GitHub.
2. LaTeX tabular snippets for the IOAI 2027 Scientific Task Proposal paper.
3. Machine-readable JSON exports for leaderboard platforms and grading servers.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from shift.diagnostics import DiagnosticReport


class ReportGenerator:
    """Generates comprehensive, publication-ready benchmark reports."""

    def __init__(self, title: str = "Project SHIFT Benchmark Evaluation Report"):
        self.title = title
        self.model_results: Dict[str, Dict[str, Any]] = {}
        self.diagnostic_reports: Dict[str, DiagnosticReport] = {}
        self.audit_results: Dict[str, Dict[str, Any]] = {}

    def add_model_results(
        self,
        model_name: str,
        results: Dict[str, Any],
        diagnostic_report: Optional[DiagnosticReport] = None
    ) -> None:
        """Registers evaluation results for a model."""
        self.model_results[model_name] = results
        if diagnostic_report:
            self.diagnostic_reports[model_name] = diagnostic_report

    def add_audit_results(
        self,
        model_name: str,
        audit_results: Dict[str, Any]
    ) -> None:
        """Registers shortcut audit results for a model."""
        self.audit_results[model_name] = audit_results

    def generate_markdown(self) -> str:
        """Generates a comprehensive Markdown report."""
        lines = [
            f"# {self.title}",
            "",
            "**Benchmark Version**: Project SHIFT v0.3 (IOAI 2027 Scientific Benchmark Infrastructure)  ",
            "**Target Domain**: In-Context Semantic Adaptation & Drift Disentanglement  ",
            "",
            "---",
            "",
            "## 1. Overall Model Performance Comparison",
            "",
            "The table below compares baseline models on exact event-frame accuracy, the official IOAI composite score (partial credit), semantic drift disentanglement, reference resolution, and counterfactual consistency.",
            "",
            "| Model Architecture | Exact Match (%) | IOAI Composite Score | Disentanglement (%) | Reference Res. (%) | Counterfactual Consist. (%) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |"
        ]

        # Sort models by composite score descending
        sorted_models = sorted(
            self.model_results.keys(),
            key=lambda m: self.model_results[m].get("composite_competition_score", 0.0),
            reverse=True
        )

        for name in sorted_models:
            res = self.model_results[name]
            em = res.get("exact_match_accuracy", 0.0)
            comp = res.get("composite_competition_score", 0.0)
            dis = res.get("disentanglement_accuracy", 0.0)
            ref = res.get("reference_resolution_accuracy", 0.0)
            cf = res.get("counterfactual_consistency", {}).get("consistency_score", 0.0)
            lines.append(f"| **{name}** | {em:.2f}% | **{comp:.2f}** | {dis:.2f}% | {ref:.2f}% | {cf:.2f}% |")

        # Slot-Level F1 Table
        lines.extend([
            "",
            "## 2. Slot-Level Macro F1 Breakdown",
            "",
            "Macro F1 scores across individual semantic frame slots:",
            "",
            "| Model Architecture | Agent F1 | Action F1 | Object F1 | Tense F1 | Polarity F1 | Mean Slot F1 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
        ])

        for name in sorted_models:
            res = self.model_results[name]
            f1_dict = res.get("slot_macro_f1", {})
            ag_f1 = f1_dict.get("agent", {}).get("macro_f1", 0.0)
            ac_f1 = f1_dict.get("action", {}).get("macro_f1", 0.0)
            ob_f1 = f1_dict.get("object", {}).get("macro_f1", 0.0)
            te_f1 = f1_dict.get("tense", {}).get("macro_f1", 0.0)
            po_f1 = f1_dict.get("polarity", {}).get("macro_f1", 0.0)
            mean_f1 = (ag_f1 + ac_f1 + ob_f1 + te_f1 + po_f1) / 5.0
            lines.append(f"| {name} | {ag_f1:.2f} | {ac_f1:.2f} | {ob_f1:.2f} | {te_f1:.2f} | {po_f1:.2f} | **{mean_f1:.2f}** |")

        # Progressive Difficulty Levels Table
        lines.extend([
            "",
            "## 3. Progressive Difficulty Levels (Curriculum Evaluation)",
            "",
            "- **Level 1**: Direct Canonical Observation (Isolated context, fixed syntax)  ",
            "- **Level 2**: Unseen Semantic Drift (Factorial community/historical shift)  ",
            "- **Level 3**: Multi-Turn Dialogue & Coreference (Anaphora resolution)  ",
            "- **Level 4**: Conversational Repair & Ellipsis (Discourse state updates)  ",
            "- **Level 5**: Counterfactual Contrastive Drift (Disentangling causal factors)  ",
            "",
            "| Model Architecture | Level 1 (Direct) | Level 2 (Drift) | Level 3 (Coreference) | Level 4 (Repair/Ellipsis) | Level 5 (Counterfactual) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |"
        ])

        for name in sorted_models:
            res = self.model_results[name]
            prog = res.get("progressive_levels", {})
            l1 = prog.get("Level_1", {}).get("accuracy", 0.0)
            l2 = prog.get("Level_2", {}).get("accuracy", 0.0)
            l3 = prog.get("Level_3", {}).get("accuracy", 0.0)
            l4 = prog.get("Level_4", {}).get("accuracy", 0.0)
            l5 = prog.get("Level_5", {}).get("accuracy", 0.0)
            lines.append(f"| {name} | {l1:.1f}% | {l2:.1f}% | {l3:.1f}% | {l4:.1f}% | {l5:.1f}% |")

        # Shortcut Audits Table (if available)
        if self.audit_results:
            lines.extend([
                "",
                "## 4. Adversarial Shortcut Audits & Leakage Resistance",
                "",
                "Evaluation of model degradation under adversarial ablations to confirm genuine in-context adaptation:",
                "",
                "| Model Architecture | Full Context | Support Ablation (k=0) | Context Ablation | Metadata Only | Permutation Drop | Vulnerability |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
            ])
            for name, audit in self.audit_results.items():
                clean = audit.get("clean_accuracy", 0.0)
                k0 = audit.get("support_ablation_accuracy", 0.0)
                ctx = audit.get("context_ablation_accuracy", 0.0)
                meta = audit.get("metadata_only_accuracy", 0.0)
                perm = audit.get("permutation_drop", 0.0)
                vuln = "Vulnerable (Shortcut)" if (k0 > 30.0 or meta > 25.0) else "Robust (Adapting)"
                lines.append(f"| {name} | {clean:.1f}% | {k0:.1f}% | {ctx:.1f}% | {meta:.1f}% | {perm:.1f}% | **{vuln}** |")

        # Diagnostics & Pedagogical Feedback
        if self.diagnostic_reports:
            lines.extend([
                "",
                "## 5. Model Diagnostics & Error Taxonomy",
                ""
            ])
            for name, diag in self.diagnostic_reports.items():
                lines.extend([
                    f"### Diagnostic Profile: {name}",
                    f"- **Total Errors**: {diag.total_errors} / {diag.total_episodes} ({100.0 - diag.exact_match_accuracy:.1f}% error rate)",
                    "- **Primary Failure Modes**:"
                ])
                for mode, count in sorted(diag.failure_counts.items(), key=lambda x: -x[1]):
                    pct = diag.failure_percentages.get(mode, 0.0)
                    lines.append(f"  - `{mode}`: {count} ({pct:.1f}% of errors)")
                lines.append("- **Recommendations for Improvement**:")
                for rec in diag.recommendations:
                    lines.append(f"  - {rec}")
                lines.append("")

        lines.append("---  \n*Report automatically generated by `shift.evaluation_report.ReportGenerator`.*")
        return "\n".join(lines)

    def generate_latex(self) -> str:
        """Generates LaTeX table for the IOAI Task Proposal paper."""
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{Project SHIFT Benchmark: Baseline Evaluation and In-Context Adaptation Performance.}",
            r"\label{tab:shift_results}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"\textbf{Model} & \textbf{Exact Match (\%)} & \textbf{Composite Score} & \textbf{Disentangle (\%)} & \textbf{Ref. Res. (\%)} & \textbf{CF Consist. (\%)} & \textbf{Mean Slot F1} \\",
            r"\midrule"
        ]

        sorted_models = sorted(
            self.model_results.keys(),
            key=lambda m: self.model_results[m].get("composite_competition_score", 0.0),
            reverse=True
        )

        for i, name in enumerate(sorted_models):
            res = self.model_results[name]
            em = res.get("exact_match_accuracy", 0.0)
            comp = res.get("composite_competition_score", 0.0)
            dis = res.get("disentanglement_accuracy", 0.0)
            ref = res.get("reference_resolution_accuracy", 0.0)
            cf = res.get("counterfactual_consistency", {}).get("consistency_score", 0.0)
            f1_dict = res.get("slot_macro_f1", {})
            slot_vals = [f1_dict.get(s, {}).get("macro_f1", 0.0) for s in ["agent", "action", "object", "tense", "polarity"]]
            mean_f1 = sum(slot_vals) / len(slot_vals) if slot_vals else 0.0

            if i == 0:
                # Top model in bold
                lines.append(
                    f"\\textbf{{{name}}} & \\textbf{{{em:.1f}}} & \\textbf{{{comp:.1f}}} & \\textbf{{{dis:.1f}}} & \\textbf{{{ref:.1f}}} & \\textbf{{{cf:.1f}}} & \\textbf{{{mean_f1:.1f}}} \\\\"
                )
            else:
                lines.append(
                    f"{name} & {em:.1f} & {comp:.1f} & {dis:.1f} & {ref:.1f} & {cf:.1f} & {mean_f1:.1f} \\\\"
                )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}"
        ])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes all results to structured dictionary."""
        return {
            "title": self.title,
            "models": self.model_results,
            "audits": self.audit_results,
            "diagnostics": {
                name: diag.to_dict() for name, diag in self.diagnostic_reports.items()
            }
        }

    def export_all(
        self,
        output_dir: str | Path,
        base_name: str = "shift_evaluation_report"
    ) -> Dict[str, Path]:
        """Exports Markdown, LaTeX, and JSON report files to directory."""
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        md_path = out_p / f"{base_name}.md"
        tex_path = out_p / f"{base_name}.tex"
        json_path = out_p / f"{base_name}.json"

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.generate_markdown())

        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(self.generate_latex())

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        return {
            "markdown": md_path,
            "latex": tex_path,
            "json": json_path
        }
