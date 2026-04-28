# =============================================
# 📊 METRICS CALCULATION FUNCTIONS
# =============================================

import os
import csv
import math
from typing import List, Dict, Any, Optional


def calculate_dcg(scores: List[float], k: int = 10) -> float:
    """
    Calculate Discounted Cumulative Gain (DCG).
    
    DCG_k = sum(rel_i / log2(i+1)) for i=1 to k
    
    Args:
        scores: List of relevance scores in ranked order
        k: Position cutoff (default 10)
    
    Returns:
        DCG value
    """
    dcg = 0.0
    for i, score in enumerate(scores[:k], start=1):
        dcg += float(score) / math.log2(i + 1)
    return dcg


def calculate_ndcg(actual_scores: List[float], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG).
    
    NDCG_k = DCG_k / IDCG_k
    where IDCG is DCG of the ideal (perfectly sorted) ranking
    
    Args:
        actual_scores: List of relevance scores in API-ranked order
        k: Position cutoff (default 10)
    
    Returns:
        NDCG value between 0 and 1
    """
    if not actual_scores:
        return 0.0
    
    # DCG of actual ranking
    dcg = calculate_dcg(actual_scores, k)
    
    # IDCG: perfect ranking (scores sorted descending)
    ideal_scores = sorted(actual_scores, reverse=True)
    idcg = calculate_dcg(ideal_scores, k)
    
    # NDCG = DCG / IDCG
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_precision_recall_metrics(
    results_dir: str = 'category_search_results_v1',
    output_csv_path: str = "performance_metrics_v1.csv",
    base_path: str = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate Precision@1 and Recall@5 for each category using Gemini-scored Top 5 as ground truth.
    
    Args:
        results_dir: Directory with scored results CSVs (e.g., category_search_results_v1/)
        output_csv_path: Path to export metrics CSV
    
    Returns:
        Dict with metrics per category: {category: {precision@1, recall@5, keyword_count}}
    
    Metrics:
        - Binary relevance: gemini score > 2 maps to 1, else 0
        - Precision@1: 1 if top result is in the Gemini Top 5 GT set, 0 otherwise
        - Precision@2: proportion of GT top results appearing in the API top‑2 slot
        - Recall@5: fraction of GT set found in API top‑5
        - Recall@20: fraction of GT set found in API top‑20

    Ground Truth: Uses _ranked_top5.csv files (Gemini-scored top 5 results per keyword).  Only those with score>2 are treated as relevant.
    """
    
    if base_path:
        results_dir = f'{base_path}/{results_dir}'
        output_csv_path = f'{base_path}/{output_csv_path}'


    if not os.path.exists(results_dir):
        print(f"⚠️ Results directory not found: {results_dir}")
        return {}
    
    # Find all category CSV files
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('_all_scored.csv')]
    print(f"📂 Found {len(csv_files)} category result files in '{results_dir}'")
    
    if not csv_files:
        print(f"⚠️ No _all_scored.csv files found in {results_dir}")
        return {}
    
    # Process each category
    categories_metrics = {}
    all_metrics_rows = []
    
    for csv_file in csv_files:
        category_name = csv_file.replace('_all_scored.csv', '')
        
        all_scored_csv_path = os.path.join(results_dir, csv_file)

        print(f"\n📊 Calculating metrics for category: {category_name}")

        # Load all scored results (this file will also be used to build ground truth)
        with open(all_scored_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_scored_data = list(reader)

        # Group by keyword
        scored_by_keyword = {}
        for row in all_scored_data:
            kw = row.get("keyword", "")
            if kw and row.get("id") != "count":
                if kw not in scored_by_keyword:
                    scored_by_keyword[kw] = []
                scored_by_keyword[kw].append(row)
        
        # NOTE: We now use the full _all_scored.csv sorted by `score` as the
        # ground-truth ranking. For each keyword we'll sort its results by
        # numeric `score` (descending) and take the top-k slices as the GT.
        
        # Calculate metrics per keyword
        category_p1_scores = []
        category_r6_scores = []
        category_metrics = {}
        
        for keyword in scored_by_keyword.keys():
            # API results (the same file contains the scored ranking we'll treat
            # as ground truth). `all_results` are the API-ranked outputs for this keyword.
            all_results = scored_by_keyword[keyword]

            # Build ground truth by sorting all results by `score` (numeric desc)
            try:
                gt_sorted = sorted(
                    [r for r in all_results if r.get("id") != "count"],
                    key=lambda x: float(x.get("score") or 0),
                    reverse=True,
                )
            except Exception:
                gt_sorted = list(all_results)

            if not gt_sorted:
                print(f"  ⚠️ No scored results to build GT for keyword: {keyword}")
                continue

            # Ground-truth ID lists for top-k comparisons
            gt_top1 = [r.get("id") for r in gt_sorted[:min(1, len(gt_sorted))] if r.get("id")]
            gt_top2 = {r.get("id") for r in gt_sorted[:min(2, len(gt_sorted))] if r.get("id")}
            gt_top6 = [r.get("id") for r in gt_sorted[:min(6, len(gt_sorted))] if r.get("id")]
            gt_top20 = {r.get("id") for r in gt_sorted[:min(20, len(gt_sorted))] if r.get("id")}

            # For reporting counts (use top-6 as GT denominator for recall@6)
            gt_ids_count = len(gt_top6)

            # Precision@1: exact match of API top-1 to GT top-1
            precision_at_1 = 0
            if all_results:
                top_result_id = all_results[0].get("id")
                if gt_top1 and top_result_id == gt_top1[0]:
                    precision_at_1 = 1

            # Precision@2: fraction of API top-2 appearing in GT top-2
            precision_at_2 = 0
            if all_results:
                top2_ids = [r.get("id") for r in all_results[:min(2, len(all_results))] if r.get("id")]
                if top2_ids:
                    precision_at_2 = sum(1 for tid in top2_ids if tid in gt_top2) / len(top2_ids)

            # Recall@6: fraction of GT top-6 that appear in API top-6
            recall_at_6 = 0
            relevant_found6 = 0
            if gt_top6:
                api_top_6_ids = {r.get("id") for r in all_results[:min(6, len(all_results))] if r.get("id")}
                relevant_found6 = len(api_top_6_ids & set(gt_top6))
                recall_at_6 = relevant_found6 / len(gt_top6)

            # Recall@20: fraction of GT top-20 that appear in API top-20
            recall_at_20 = 0
            relevant_found20 = 0
            if gt_top20:
                api_top_20_ids = {r.get("id") for r in all_results[:min(20, len(all_results))] if r.get("id")}
                relevant_found20 = len(api_top_20_ids & gt_top20)
                recall_at_20 = relevant_found20 / len(gt_top20)
            
            # NDCG@10: Normalized Discounted Cumulative Gain at position 10
            # Uses numeric scores from all_results as relevance signal
            ndcg_at_10 = 0.0
            if all_results:
                scores_for_ndcg = [float(r.get("score", 0)) for r in all_results[:min(10, len(all_results))] if r.get("id") != "count"]
                if scores_for_ndcg:
                    ndcg_at_10 = calculate_ndcg(scores_for_ndcg, k=10)
            
            # Build numerator/denominator info for each metric
            # P@1
            p1_num = 1 if precision_at_1 == 1 else 0
            p1_den = 1
            # P@2
            top2_ids = [r.get("id") for r in all_results[:min(2, len(all_results))] if r.get("id")]
            p2_den = len(top2_ids)
            p2_num = sum(1 for tid in top2_ids if tid in gt_top2) if p2_den else 0
            # R@6
            r6_num = relevant_found6
            r6_den = len(gt_top6)
            # R@20
            r20_num = relevant_found20
            r20_den = len(gt_top20)

            # Store metrics (include metric_counts string)
            metric_counts_str = f"P@1:{p1_num}/{p1_den};P@2:{p2_num}/{p2_den};R@6:{r6_num}/{r6_den};R@20:{r20_num}/{r20_den};NDCG@10:{ndcg_at_10:.3f}"
            category_metrics[keyword] = {
                "precision@1": precision_at_1,
                "precision@2": precision_at_2,
                "recall@6": recall_at_6,
                "recall@20": recall_at_20,
                "ndcg@10": ndcg_at_10,
                "all_results_count": len(all_results),
                "gt_count": gt_ids_count,
                "metric_counts": metric_counts_str
            }
            category_p1_scores.append(precision_at_1)
            category_p2_scores = category_metrics.setdefault("_p2_list", [])
            category_p2_scores.append(precision_at_2)
            category_r6_scores.append(recall_at_6)
            category_r20_scores = category_metrics.setdefault("_r20_list", [])
            category_r20_scores.append(recall_at_20)
            category_ndcg_scores = category_metrics.setdefault("_ndcg_list", [])
            category_ndcg_scores.append(ndcg_at_10)
            
            print(f"  ✓ {keyword}: P@1={precision_at_1:.0%}, P@2={precision_at_2:.0%}, R@6={recall_at_6:.1%}, R@20={recall_at_20:.1%}, NDCG@10={ndcg_at_10:.3f} ({len(all_results)} results, {gt_ids_count} GT top-6)")
        
        # Calculate category averages
        avg_p1 = sum(category_p1_scores) / len(category_p1_scores) if category_p1_scores else 0
        avg_p2 = (sum(category_metrics.get("_p2_list", [])) / len(category_metrics.get("_p2_list", []))) if category_metrics.get("_p2_list") else 0
        avg_r6 = sum(category_r6_scores) / len(category_r6_scores) if category_r6_scores else 0
        avg_r20 = (sum(category_metrics.get("_r20_list", [])) / len(category_metrics.get("_r20_list", []))) if category_metrics.get("_r20_list") else 0
        avg_ndcg = (sum(category_metrics.get("_ndcg_list", [])) / len(category_metrics.get("_ndcg_list", []))) if category_metrics.get("_ndcg_list") else 0
        
        del category_metrics["_p2_list"]
        del category_metrics["_r20_list"]
        del category_metrics["_ndcg_list"]
        categories_metrics[category_name] = {
            "precision@1": avg_p1,
            "precision@2": avg_p2,
            "recall@6": avg_r6,
            "recall@20": avg_r20,
            "ndcg@10": avg_ndcg,
            "keyword_count": len(category_metrics)
        }
        
        print(f"  📈 Category averages: P@1={avg_p1:.1%}, P@2={avg_p2:.1%}, R@6={avg_r6:.1%}, R@20={avg_r20:.1%}, NDCG@10={avg_ndcg:.3f} ({len(category_metrics)} keywords)")
        # Prepare rows for export
        for keyword, metrics in category_metrics.items():
            all_metrics_rows.append({
                "category": category_name,
                "keyword": keyword,
                "precision@1": f"{metrics['precision@1']:.0%}",
                "precision@2": f"{metrics['precision@2']:.0%}",
                "recall@6": f"{metrics['recall@6']:.1%}",
                "recall@20": f"{metrics['recall@20']:.1%}",
                "ndcg@10": f"{metrics['ndcg@10']:.3f}",
                "all_results_count": metrics["all_results_count"],
                "gt_top6_count": metrics["gt_count"],
                "metric_counts": metrics.get("metric_counts", "")
            })
    
    # Export metrics to CSV
    if all_metrics_rows:
        os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
        
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "category", "keyword", "precision@1", "precision@2",
                "recall@6", "recall@20", "ndcg@10", "all_results_count", "gt_top6_count", "metric_counts"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics_rows)
        
        print(f"\n🎉 Exported {len(all_metrics_rows)} metric rows to '{output_csv_path}'")
    
    # Export category summary
    if categories_metrics:
        summary_path = output_csv_path.replace('.csv', '_summary.csv')
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "category", "precision@1", "precision@2",
                "recall@6", "recall@20", "ndcg@10", "keyword_count"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for cat, metrics in categories_metrics.items():
                writer.writerow({
                    "category": cat,
                    "precision@1": f"{metrics['precision@1']:.1%}",
                    "precision@2": f"{metrics['precision@2']:.1%}",
                    "recall@6": f"{metrics['recall@6']:.1%}",
                    "recall@20": f"{metrics['recall@20']:.1%}",
                    "ndcg@10": f"{metrics['ndcg@10']:.3f}",
                    "keyword_count": metrics["keyword_count"]
                })
        
        print(f"🎉 Exported category summary to '{summary_path}'")
    
    return categories_metrics


calculate_precision_recall_metrics()
