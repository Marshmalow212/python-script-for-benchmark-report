"""
Enhanced Search Report Score with Gemini AI Scoring
=====================================================

Pipeline: Search API → Results → Gemini Scoring → Aggregated Results

This module extends search_report_score.py with Gemini-based result scoring.
Each search result is evaluated by Gemini AI before being exported to CSV.

Usage:
    python3 search_report_score_enhanced_gemini.py brand TRUE tel_type postpaid lang th
"""

import os
import json
import requests
import csv
import time
import sys
from typing import List, Dict, Any, Optional
import re

# =============================================
# CONFIGURATION - OVERRIDABLE VIA CLI
# =============================================
lang = 'th'
brand = 'guest'
tel_type = 'guest'
COUNT = 100
TOKEN_REFRESH_INTERVAL = 900
PAGE = None
LIMIT = 10

# Gemini Configuration (HTTP API)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyDnQ_Va3Z9m8hYlVyNlAwhKZmoi4Kaouzg')
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent'
GEMINI_TIMEOUT = 30  # seconds

# API Configuration
# SEARCH_API_URL = "https://search.true.th/api/v1/search"  # Can be overridden
SEARCH_API_URL = "https://search-uat.true.th/api/v1/search"  # Can be overridden
# Alternative endpoints: https://search-uat.true.th/api/v1/search (uat)

last_token_time = None
raw_results = []


# =============================================
# 📡 SEARCH API FUNCTIONS
# =============================================

def get_access_token():
    """
    Retrieve OAuth token from TrueApp Common API.
    Valid for ~900 seconds.
    """
    env = 'uat'  # Change as needed: 'uat' or 'preprod'
    token_conf = {
        "uat": {
            "url": "https://trueapp-commonapi-uat.true.th/authen/v1/token/request",
            "payload": json.dumps({
                "clientId": "trueappuat",
                "clientSecret": "appsecretuat"
            })
        },
        "preprod": {
            "url": "https://trueapp-commonapi-preprod.true.th/authen/v1/token/request",
            "payload": json.dumps({
                "clientId": "trueapppreprod",
                "clientSecret": "appsecrepreprod"
            })
        }
    }
    headers = {
        'sourceSystemId': 'TRUEAPP',
        'sessionId': '21A75111822',
        'deviceId': 'd1',
        'platform': 'ios',
        'language': 'EN',
        'version': '1.0.0',
        'Content-Type': 'application/json',
    }

    try:
        response = requests.post(token_conf[env]["url"], headers=headers, data=token_conf[env]["payload"])
        data = response.json()
        access_token = data.get('data', {}).get('accessToken', None)
        if access_token:
            return access_token
    except Exception as e:
        print(f"❌ Error obtaining access token: {e}")
    
    return None


def call_search_api(keyword: str, b_token: Optional[str] = None, url: str = SEARCH_API_URL) -> Optional[Dict[str, Any]]:
    """
    Call TrueApp Search API for a single keyword.
    
    Args:
        keyword: Search keyword
        b_token: Bearer token (auto-fetch if None)
        url: API endpoint
    
    Returns:
        Dict with API response or None if error
    """
    headers = {
        "Content-Type": "application/json",
        "version": "1.1.0",
        "platform": "ANDROID",
        "x-user-id": "e73ab1a297d255d83782148e5bab3971",
        "x-user-brand": brand,
        "x-user-type": tel_type,
        "language": lang,
        "Authorization": f"Bearer {b_token or get_access_token()}",
    }
    
    payload = {"type": "text", "content": keyword}
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error fetching for '{keyword}': {e}")
        return None


# =============================================
# 🤖 GEMINI SCORING FUNCTIONS
# =============================================

def call_gemini_api(prompt: str, api_key: str = GEMINI_API_KEY, 
                    api_url: str = GEMINI_API_URL, timeout: int = GEMINI_TIMEOUT) -> Optional[str]:
    """
    Call Gemini API via HTTP with the provided prompt.
    
    Args:
        prompt: The prompt text to send to Gemini
        api_key: Google Gemini API key
        api_url: Gemini API endpoint URL
        timeout: Request timeout in seconds
    
    Returns:
        Response text from Gemini, or None if error
    
    Raises:
        ValueError: If API key not provided
        Exception: If API call fails
    """
    if not api_key:
        raise ValueError(
            "❌ GEMINI_API_KEY not set. "
            "Set via environment variable: export GEMINI_API_KEY='your-key'"
        )
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 5,
            "topP": 1,
            "topK": 1
        }
    }
    
    try:
        response = requests.post(
            f"{api_url}?key={api_key}",
            json=payload,
            timeout=timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
        
        json_response = response.json()
        text = json_response['candidates'][0]['content']['parts'][0]['text'].strip()
        return text
        
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return None


def build_gemini_scoring_prompt(keyword: str, search_results: List[Dict[str, Any]], 
                                custom_prompt: Optional[str] = None) -> str:
    """
    Build prompt for Gemini to score search results.
    
    Args:
        keyword: Original search keyword
        search_results: List of result dicts with id, title, description, source
        custom_prompt: Optional custom scoring instructions (provided later)
    
    Returns:
        Formatted prompt for Gemini
    """
    if custom_prompt:
        base_prompt = custom_prompt
    else:
        # Default scoring prompt - will be replaced with user's prompt later
        base_prompt = """
You are a search result relevance evaluator. Score each result from 0-10 based on relevance to the search query.
Consider:
- Title relevance to query
- Description match
- Source credibility
- Overall usefulness

Return a JSON object with result_id as key and score (0-10) as value.
Example: {"id_1": 8, "id_2": 5, "id_3": 9}
"""
    
    results_text = json.dumps(search_results, ensure_ascii=False, indent=2)
    
    prompt = f"""{base_prompt}

SEARCH KEYWORD: {keyword}

SEARCH RESULTS:
{results_text}

Provide only valid JSON output with no additional text.
"""
    return prompt


def score_results_with_gemini(keyword: str, search_results: List[Dict[str, Any]], 
                             custom_prompt: Optional[str] = None) -> Dict[str, float]:
    """
    Call Gemini API to score search results via HTTP.
    
    Args:
        keyword: Search keyword (for context)
        search_results: List of results to score
        custom_prompt: Optional custom scoring instructions
    
    Returns:
        Dict mapping result_id → score (0-3)
    
    Example:
        {"id_1": 3, "id_2": 0, "id_3": 2}
    """
    if not search_results:
        print(f"⚠️ No results to score for keyword: {keyword}")
        return {}
    
    try:
        # The provided prompt (result_scoring_prompt) is designed to score a single result
        # so we call Gemini once per result and expect a single-token numeric output (0,1,2,3)
        scores: Dict[str, float] = {}

        for res in search_results:
            rid = res.get("id")
            title = res.get("title", "") or ""
            description = res.get("description", "") or ""

            # Prefer a per-result prompt function if available
            try:
                prompt_text = result_scoring_prompt(keyword, title, description)
            except NameError:
                # Fallback to compact prompt asking for a single number
                prompt_text = f"Evaluate relevance and return ONE number (0-3) for this result.\nSearch Keyword: {keyword}\nTitle: {title}\nDescription: {description}\nReturn only the number."
            start = time.time() 
            response_text = call_gemini_api(prompt_text)
            end = time.time()
            print(f'     Gemini scored result {rid} in {end - start:.2f}s: {response_text}')
            time.sleep(2.0)  # Rate limit between Gemini calls

            if not response_text:
                scores[rid] = 0
                continue

            # Extract the first digit 0-3 from the response
            m = re.search(r"\b([0-3])\b", response_text)
            if m:
                try:
                    scores[rid] = int(m.group(1))
                except Exception:
                    scores[rid] = 0
            else:
                # If parsing fails, default to 0
                scores[rid] = 0

        print(f"✅ Gemini scored {len(scores)} results for '{keyword}'")
        return scores

    except Exception as e:
        print(f"❌ Gemini scoring error for '{keyword}': {e}")
        return {}


def merge_gemini_scores(results: List[Dict[str, Any]], gemini_scores: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Merge Gemini scores into result objects.
    
    Args:
        results: Original result dicts with 'id' field
        gemini_scores: Dict mapping id → score from Gemini
    
    Returns:
        Results with new 'gemini_score' key added
    """
    for result in results:
        result_id = result.get('id')
        if result_id in gemini_scores:
            result['gemini_score'] = gemini_scores[result_id]
        else:
            result['gemini_score'] = 0.0  # Default if not scored
    
    return results


# =============================================
# 🔄 ENHANCED MAIN FUNCTION
# =============================================

def fetch_and_save_results_from_json_with_gemini(
    json_path: str,
    gt_path: str,
    url: str = SEARCH_API_URL,
    b_token: Optional[str] = None,
    output_csv: str = "results.csv",
    gemini_scoring_prompt: Optional[str] = None,
    enable_gemini: bool = True
):
    """
    Enhanced version: Search API → Results → Gemini Scoring → CSV Export
    
    Pipeline:
    1. Load keywords from JSON (by category)
    2. For each keyword:
       a. Call search API
       b. Extract top results
       c. (Optional) Score results with Gemini
       d. Merge Gemini scores into result objects
    3. Export per-category CSVs with Gemini scores
    
    Args:
        json_path: Keywords JSON file (nested by category)
        gt_path: Ground truth JSON file (for compatibility, not used here)
        url: Search API endpoint
        b_token: Bearer token (auto-fetch if None)
        output_csv: Output CSV filename pattern
        gemini_scoring_prompt: Custom prompt for Gemini scoring
        enable_gemini: Whether to enable Gemini scoring (default: True)
    
    Output:
        Per-category CSV files in category_search_results/ with columns:
        [keyword, id, title, description, source, brand, tel_type, applink, 
         score (API score), current_index, gemini_score]
    """
    
    # Validate Gemini API key if enabled
    if enable_gemini:
        if not GEMINI_API_KEY:
            print(f"⚠️ GEMINI_API_KEY not set. Continuing without Gemini scoring...")
            enable_gemini = False
        else:
            print("✅ Gemini API key detected - HTTP API ready")
    
    # Load keywords
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ JSON file not found at {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, list):
        total = LIMIT or len(data)
        chunk = 0 if not PAGE else PAGE
        start = int(chunk) * int(total)
        words = data[start: (int(start)+int(total))]
        print(f"✅ Loaded {len(words)} keywords (list format)")
    elif isinstance(data, dict):
        words = data
        print(f"✅ Loaded {len(words)} categories with keywords")
    else:
        raise ValueError("❌ Invalid JSON format. Must be a list or dict.")
    
    # Process each category
    for cat, keywords in words.items():
        results = []  # All scored results in original order (for metrics calculation)
        sorted_results = []  # All scored results sorted by Gemini score (for analysis)
        ranked_results = []  # Top 5 ranked results (for comparison)
        gemini_batch_count = 0
        
        print(f"\n📂 Processing category: {cat}")
        
        for word in keywords:
            if not word.get('content', None):
                continue
            
            content = word.get('content')
            
            # ===== STEP 1: SEARCH =====
            api_response = call_search_api(content, b_token, url)
            if not api_response:
                continue
            all_items = api_response.get("data", {}).get("items", [])
            print(f"   API returned {len(all_items)} results for '{content}'")
            # select up to 50 results from API for scoring/metrics
            items = all_items[:min(50, len(all_items))]
            if not items:
                print(f"⚠️ No items found for keyword '{content}'")
                continue
            
            # ===== STEP 2: EXTRACT RESULTS =====
            # Score all results (not limited to 10)
            all_items_count = len(items)
            print(f"   Selected {all_items_count} results to score")
            
            # ===== STEP 3: GEMINI SCORING (Optional) =====
            if enable_gemini:
                # Build clean result objects for Gemini
                clean_results = [
                    {
                        "id": item.get("id"),
                        "title": item.get("title"),
                        "description": item.get("description"),
                        "source": item.get("source")
                    }
                    for item in items
                ]
                
                gemini_scores = score_results_with_gemini(
                    content, 
                    clean_results,
                    gemini_scoring_prompt
                )
                gemini_batch_count += 1
                time.sleep(20.0)
                
                # Add rate limiting for Gemini API (1-2 requests/second is safe)
                time.sleep(0.5)
            else:
                gemini_scores = {}
            
            # ===== STEP 4: MERGE SCORES & SORT BY GEMINI SCORE (DESCENDING) =====
            # Create result dicts with Gemini scores
            scored_items = []
            for item in items:
                api_score_val = item.get("score")
                gem_score = gemini_scores.get(item.get("id"), 0) if enable_gemini else 0

                result_dict = {
                    "keyword": content,
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "description": item.get("description"),
                    "source": item.get("source"),
                    "brand": brand,
                    "tel_type": tel_type,
                    "applink": item.get("applink"),
                    "api_score": api_score_val,  # original API score
                    "score": gem_score,         # Gemini-assigned score (0-3)
                    "current_index": None,  # Will be set after sorting
                }
                scored_items.append(result_dict)
            
            # Add ALL scored results to main results list in ORIGINAL ORDER (for metrics calculation)
            for idx, item in enumerate(scored_items, start=1):
                item["current_index"] = idx  # sequential index for all results
                results.append(item.copy())  # Use copy to maintain separate lists
            
            # Also add to sorted list (will be sorted later)
            for item in scored_items:
                sorted_results.append(item.copy())
            
            # Add count summary row for all results
            results.append({
                "keyword": content,
                "id": "count",
                "title": f"All {all_items_count}",
                "description": "",
                "source": "",
                "brand": "",
                "tel_type": "",
                "applink": "",
                "api_score": "",
                "score": "",
                "current_index": "",
            })
            
            # Sort scored_items by Gemini score descending for ranking
            scored_items.sort(key=lambda x: x["score"], reverse=True)
            
            # Keep 50 ranked results (separate list)
            top_5_items = scored_items
            
            # Set current_index based on final ranking (1-based) for top 5
            for idx, item in enumerate(top_5_items, start=1):
                item["current_index"] = idx
                ranked_results.append(item)
            
            # Add count summary row for ranked results
            ranked_results.append({
                "keyword": content,
                "id": "count",
                "title": f"{len(top_5_items)} of {all_items_count}",
                "description": "",
                "source": "",
                "brand": "",
                "tel_type": "",
                "applink": "",
                "api_score": "",
                "score": "",
                "current_index": "",
            })
            
            print(f"✅ Processed: '{content}' → Top 5 of {all_items_count} results (Gemini: {enable_gemini})")
            
            # Rate limit search API requests
            time.sleep(0.3)
        
        # ===== STEP 5: EXPORT CSV =====
        # Export all three: original order + sorted + top 5 ranked results
        if results:
            fieldnames = [
                "keyword", "id", "title", "description", "source",
                "brand", "tel_type", "applink", "api_score", "score", "current_index"
            ]
            
            os.makedirs('category_search_results_v1', exist_ok=True)
            
            # Export ALL scored results in ORIGINAL ORDER (for metrics calculation)
            output_path_all = f'category_search_results_v1/{cat}_all_scored.csv'
            with open(output_path_all, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            result_rows = [r for r in results if r["id"] != "count"]
            print(f"🎉 Exported {len(results)} rows ({len(result_rows)} results + {len(results) - len(result_rows)} summaries) to '{output_path_all}'")
            
            # Export ALL scored results SORTED BY GEMINI SCORE (for analysis)
            if sorted_results:
                # Sort the copied results by score descending
                sorted_results_with_idx = []
                for idx, item in enumerate(sorted_results, start=1):
                    item_copy = item.copy()
                    item_copy["current_index"] = idx
                    sorted_results_with_idx.append(item_copy)
                
                # Add count summary row for sorted results
                sorted_results_with_idx.append({
                    "keyword": "",
                    "id": "count",
                    "title": f"All {len([r for r in sorted_results_with_idx if r['id'] != 'count'])}",
                    "description": "",
                    "source": "",
                    "brand": "",
                    "tel_type": "",
                    "applink": "",
                    "api_score": "",
                    "score": "",
                    "current_index": "",
                })
                
                output_path_sorted = f'category_search_results_v1/{cat}_all_sorted.csv'
                with open(output_path_sorted, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(sorted_results_with_idx)
                
                sorted_rows = [r for r in sorted_results_with_idx if r["id"] != "count"]
                print(f"🎉 Exported {len(sorted_results_with_idx)} rows ({len(sorted_rows)} results sorted + summaries) to '{output_path_sorted}'")
            
            # Export TOP 5 ranked results (for comparison)
            if ranked_results:
                output_path_ranked = f'category_search_results_v1/{cat}_ranked_top5.csv'
                with open(output_path_ranked, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(ranked_results)
                
                ranked_rows = [r for r in ranked_results if r["id"] != "count"]
                print(f"🎉 Exported {len(ranked_results)} rows ({len(ranked_rows)} top-5 results + {len(ranked_results) - len(ranked_rows)} summaries) to '{output_path_ranked}'")
            
            print(f"   Gemini batches processed: {gemini_batch_count}")
        else:
            print(f"⚠️ No results found for category '{cat}'")


# =============================================
# 📋 HELPER: SET CUSTOM GEMINI PROMPT
# =============================================

def set_gemini_prompt(prompt: str) -> str:
    """
    Store custom Gemini scoring prompt for later use.
    
    Example:
        custom_prompt = '''
        You are a telecom product evaluator. Score results based on:
        - Relevance to TRUE mobile packages
        - Data benefits clarity
        - Price competitiveness
        Return JSON with id → score.
        '''
        set_gemini_prompt(custom_prompt)
    
    Args:
        prompt: Custom prompt text
    
    Returns:
        Stored prompt (for confirmation)
    """
    global GEMINI_SCORING_PROMPT
    GEMINI_SCORING_PROMPT = prompt
    print(f"✅ Custom Gemini prompt set ({len(prompt)} chars)")
    return prompt


# =============================================
# 📊 METRICS CALCULATION FUNCTIONS
# =============================================

def calculate_precision_recall_metrics(
    results_dir: str = 'category_search_results_v1',
    output_csv_path: str = "performance_metrics_v1.csv"
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
        
        # Load the Gemini-scored Top 5 as ground truth
        ranked_csv_path = os.path.join(results_dir, f'{category_name}_ranked_top5.csv')
        all_scored_csv_path = os.path.join(results_dir, csv_file)
        
        if not os.path.exists(ranked_csv_path):
            print(f"⚠️ Ranked file not found: {ranked_csv_path}")
            continue
        
        print(f"\n📊 Calculating metrics for category: {category_name}")
        
        # Load all scored results
        with open(all_scored_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_scored_data = list(reader)
        
        # Load top 5 ranked results (this is our ground truth)
        with open(ranked_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            ranked_data = list(reader)
        
        # Group by keyword
        scored_by_keyword = {}
        for row in all_scored_data:
            kw = row.get("keyword", "")
            if kw and row.get("id") != "count":
                if kw not in scored_by_keyword:
                    scored_by_keyword[kw] = []
                scored_by_keyword[kw].append(row)
        
        ranked_by_keyword = {}
        for row in ranked_data:
            kw = row.get("keyword", "")
            if kw and row.get("id") != "count":
                if kw not in ranked_by_keyword:
                    ranked_by_keyword[kw] = []
                ranked_by_keyword[kw].append(row)
        
        # Calculate metrics per keyword
        category_p1_scores = []
        category_r5_scores = []
        category_metrics = {}
        
        for keyword in scored_by_keyword.keys():
            all_results = scored_by_keyword[keyword]
            gt_results = ranked_by_keyword.get(keyword, [])  # Top 5 is ground truth
            
            if not gt_results:
                print(f"  ⚠️ No ranked results for keyword: {keyword}")
                continue
            
            # Ground truth: Set of result IDs from Gemini Top 5 with score > 2
            gt_ids = {row.get("id") for row in gt_results
                      if row.get("id") and row.get("score") and float(row.get("score")) > 2}
            
            # Binary relevance mapping for all API results (score>2 => 1)
            binary_rels = [1 if (r.get("score") and float(r.get("score")) > 2) else 0
                           for r in all_results]

            # Precision@1: is the very first API hit relevant and in GT?
            precision_at_1 = 0
            if all_results:
                top_result_id = all_results[0].get("id")
                if top_result_id in gt_ids and binary_rels[0] == 1:
                    precision_at_1 = 1

            # Precision@2: fraction of the first two hits that are in GT
            precision_at_2 = 0
            if all_results:
                top2_ids = [r.get("id") for r in all_results[:2] if r.get("id")]
                if top2_ids:
                    precision_at_2 = sum(1 for tid in top2_ids if tid in gt_ids) / len(top2_ids)

            # Recall@5: what % of GT set appears in API top-5
            recall_at_5 = 0
            if gt_ids:
                api_top_5_ids = {r.get("id") for r in all_results[:5] if r.get("id")}
                relevant_found = len(api_top_5_ids & gt_ids)
                recall_at_5 = relevant_found / len(gt_ids)

            # Recall@20: what % of GT set appears in API top-20
            recall_at_20 = 0
            if gt_ids:
                api_top_20_ids = {r.get("id") for r in all_results[:20] if r.get("id")}
                relevant_found20 = len(api_top_20_ids & gt_ids)
                recall_at_20 = relevant_found20 / len(gt_ids)
            
            # Store metrics
            category_metrics[keyword] = {
                "precision@1": precision_at_1,
                "precision@2": precision_at_2,
                "recall@5": recall_at_5,
                "recall@20": recall_at_20,
                "all_results_count": len(all_results),
                "gt_count": len(gt_ids)
            }
            
            category_p1_scores.append(precision_at_1)
            category_p2_scores = category_metrics.setdefault("_p2_list", [])
            category_p2_scores.append(precision_at_2)
            category_r5_scores.append(recall_at_5)
            category_r20_scores = category_metrics.setdefault("_r20_list", [])
            category_r20_scores.append(recall_at_20)
            
            print(f"  ✓ {keyword}: P@1={precision_at_1:.0%}, P@2={precision_at_2:.0%}, R@5={recall_at_5:.1%}, R@20={recall_at_20:.1%} ({len(all_results)} results, {len(gt_ids)} GT top-5)")
        
        # Calculate category averages
        avg_p1 = sum(category_p1_scores) / len(category_p1_scores) if category_p1_scores else 0
        avg_p2 = (sum(category_metrics.get("_p2_list", [])) / len(category_metrics.get("_p2_list", []))) if category_metrics.get("_p2_list") else 0
        avg_r5 = sum(category_r5_scores) / len(category_r5_scores) if category_r5_scores else 0
        avg_r20 = (sum(category_metrics.get("_r20_list", [])) / len(category_metrics.get("_r20_list", []))) if category_metrics.get("_r20_list") else 0
        
        categories_metrics[category_name] = {
            "precision@1": avg_p1,
            "precision@2": avg_p2,
            "recall@5": avg_r5,
            "recall@20": avg_r20,
            "keyword_count": len(category_metrics)
        }
        
        print(f"  📈 Category averages: P@1={avg_p1:.1%}, P@2={avg_p2:.1%}, R@5={avg_r5:.1%}, R@20={avg_r20:.1%} ({len(category_metrics)} keywords)")
        
        # Prepare rows for export
        for keyword, metrics in category_metrics.items():
            all_metrics_rows.append({
                "category": category_name,
                "keyword": keyword,
                "precision@1": f"{metrics['precision@1']:.0%}",
                "precision@2": f"{metrics['precision@2']:.0%}",
                "recall@5": f"{metrics['recall@5']:.1%}",
                "recall@20": f"{metrics['recall@20']:.1%}",
                "all_results_count": metrics["all_results_count"],
                "gt_top5_count": metrics["gt_count"]
            })
    
    # Export metrics to CSV
    if all_metrics_rows:
        os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
        
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "category", "keyword", "precision@1", "precision@2",
                "recall@5", "recall@20", "all_results_count", "gt_top5_count"
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
                "recall@5", "recall@20", "keyword_count"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for cat, metrics in categories_metrics.items():
                writer.writerow({
                    "category": cat,
                    "precision@1": f"{metrics['precision@1']:.1%}",
                    "precision@2": f"{metrics['precision@2']:.1%}",
                    "recall@5": f"{metrics['recall@5']:.1%}",
                    "recall@20": f"{metrics['recall@20']:.1%}",
                    "keyword_count": metrics["keyword_count"]
                })
        
        print(f"🎉 Exported category summary to '{summary_path}'")
    
    return categories_metrics


# =============================================
# 🧪 EXAMPLE USAGE
# =============================================

if __name__ == "__main__":
    # Example custom prompt for TrueApp domain
    CUSTOM_GEMINI_PROMPT = """
    You are a TRUE telecom product relevance evaluator. Score each search result 
    based on how well it matches the user's search intent for Thai telecom services.
    
    Scoring criteria (0-10):
    - 9-10: Exact match, high relevance (e.g., product page)
    - 7-8: Relevant content, good match
    - 5-6: Partially relevant, some utility
    - 3-4: Tangentially related
    - 0-2: Irrelevant or misleading
    
    Return ONLY a JSON object like: {"id_1": 9, "id_2": 7, "id_3": 4}
    No explanations, no extra text.
    """
    
    # Configuration
    json_path = os.path.join(os.getcwd(), "trueapp_keywords_by_category.json") # full test
    # json_path = os.path.join(os.getcwd(), "trueapp_keywords_by_category_extra.json") # small test set
    # json_path = os.path.join(os.getcwd(), "trueapp_keywords_by_category_test.json") # small test set
    gt_path = os.path.join(os.getcwd(), "keywords_gt_map_top.json")
    url = "https://search.true.th/api/v1/search"  # prod
    # url = "https://search-uat.true.th/api/v1/search"  # uat
    b_token = None
    report_path = "results_with_gemini_scores.csv"
    
    # CLI argument overrides
    if len(sys.argv) > 1:
        if 'brand' in sys.argv:
            brand = sys.argv[sys.argv.index('brand') + 1]
        if 'tel_type' in sys.argv:
            tel_type = sys.argv[sys.argv.index('tel_type') + 1]
        if 'token' in sys.argv:
            b_token = sys.argv[sys.argv.index('token') + 1]
        if 'lang' in sys.argv:
            lang = sys.argv[sys.argv.index('lang') + 1]
    
    # Run enhanced search with Gemini scoring
    try:
        fetch_and_save_results_from_json_with_gemini(
            json_path=json_path,
            gt_path=gt_path,
            url=url,
            b_token=b_token,
            output_csv=report_path,
            gemini_scoring_prompt=CUSTOM_GEMINI_PROMPT,
            enable_gemini=True
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # calculate_precision_recall_metrics() 
  


def result_scoring_prompt(keyword, title, description):
  return f"""You are a search relevance evaluator for TrueApp, the official mobile app of True Corporation (Thailand's telecom company serving TrueMove H, DTAC, TrueOnline, TrueVisions).

Your job is to think like a PRODUCT MANAGER: understand what the user is trying to accomplish when they search, then evaluate if the result helps them achieve that goal.

SCORING FRAMEWORK:
3 = PERFECT MATCH: Result is exactly what the user is looking for
2 = HELPFUL ALTERNATIVE: Result serves the same user need or is functionally related
1 = WEAK CONNECTION: Same broad category but unlikely to satisfy the user
0 = IRRELEVANT: Does not help the user achieve their goal

EVALUATION PROCESS:
Step 1: What is the user's GOAL when searching this keyword?
Step 2: Does this result help achieve that goal?
Step 3: Score based on how directly it helps

USER INTENT MAPPING:

[DEVICE SEARCHES: iPhone, iPad, Samsung, OPPO, VIVO, Apple Watch, โทรศัพท์]
User Goal: Buy/upgrade a phone or device, find device promotions
→ Score 3: Device sales, installment plans, trade-in offers for that brand
→ Score 2: Device promotions for similar category (e.g., other smartphones)
→ Score 0: Mobile packages, top-up, SIM services, merchandise/gifts, ANY non-device result

[eSIM / SIM SERVICES]
User Goal: Get a new SIM, activate eSIM, or get SIM for travel
→ Score 3: eSIM activation, eSIM setup service, SIM registration
→ Score 2: Travel SIM products (GO Travel SIM), Roaming SIM packages, International SIM - because these are often delivered/activated as eSIM
→ Score 1: General SIM-related services
→ Score 0: Gaming packages (EsportMax), data packages, anything not about SIM/eSIM itself
→ IMPORTANT: "Esport" is NOT related to "eSIM" - they are completely different words

[DATA PACKAGES: เน็ต, Internet, 5GB, Unlimited, WiFi, แพ็กเกจ, โปรเน็ต]
User Goal: Buy internet/data for their phone
→ Score 3: Data packages matching their search (size, speed, price)
→ Score 2: Related data packages or bundles with data
→ Score 0: Voice packages, privileges, devices, streaming services

[VOICE/CALL PACKAGES: โทร, นาที, Call, โทรฟรี]
User Goal: Get calling minutes or voice packages
→ Score 3: Voice/call packages
→ Score 2: Bundles that include voice minutes
→ Score 0: Data-only packages, privileges, devices

[STREAMING SERVICES: Netflix, YouTube, iQIYI, Viu, WeTV, Spotify, Disney, TrueID, TrueVisions]
User Goal: Access specific streaming content
→ Score 3: Result explicitly offers/mentions the EXACT service searched
→ Score 2: Bundle that includes the searched service
→ Score 0: Different streaming service without mentioning the searched one
→ CRITICAL: If user searches "Netflix", result must mention "Netflix" for score 2+. TrueID alone = 0

[TRAVEL/ROAMING: Roaming, Japan, ญี่ปุ่น, Singapore, โรมมิ่ง, Travel]
User Goal: Get internet/calls while traveling abroad
→ Score 3: Roaming package for searched destination, Travel SIM
→ Score 2: General roaming/travel packages, GO Travel SIM
→ Score 0: Domestic packages, unrelated services

[FOOD & BEVERAGE BRANDS: Starbucks, KFC, McDonald's, Cafe Amazon, เต่าบิน, TrueCoffee]
User Goal: Get discount/privilege at that specific brand
→ Score 3: Promotion explicitly for the searched brand
→ Score 0: Promotion for ANY different brand - users want THAT specific brand, not alternatives

[RETAIL/PARTNER BRANDS: 7-11, Lazada, Shopee, Grab, SF Cinema, Major]
User Goal: Get discount/privilege at that specific store/service
→ Score 3: Promotion for the exact brand searched
→ Score 0: Promotion for different brand - no substitutes

[GIVEAWAY/FREEBIES: แจก, Giveaway, FreeGifts, ใจดีแจกวัน, รับฟรี, True Bonus]
User Goal: Get something for FREE
→ Score 3: Free giveaways, daily rewards, free gifts
→ Score 2: Games/gamification with free rewards, point collection games
→ Score 1: Discounts (not free), cashback
→ Score 0: Paid packages, regular promotions

[JAIDEE/EMERGENCY: ใจดี, Jaidee, Emergency Refill, ยืม]
User Goal: Borrow credit/data when running low
→ Score 3: Jaidee loan service, emergency credit/data
→ Score 2: Usage info (helps decide when to borrow), balance check, account status
→ Score 0: Regular packages (user needs to borrow, not buy)

[TRUEPOINTS/REWARDS: TruePoint, ทรูพอยท์, แลกพอย, Point]
User Goal: Use or earn loyalty points
→ Score 3: Point redemption, point earning activities
→ Score 2: Promotions that give bonus points
→ Score 0: Services that don't involve points

[BILL/PAYMENT: จ่ายบิล, Bill, ชำระ]
User Goal: Pay their True/DTAC bill
→ Score 3: Bill payment service
→ Score 2: Payment promotions, cashback on bill payment
→ Score 0: Packages, privileges, unrelated services

[GENERAL PRIVILEGES: สิทธิพิเศษ, Privilege]
User Goal: Find any available discounts/benefits
→ Score 3: Privilege/discount offers
→ Score 2: Partner promotions, member benefits
→ Score 0: Regular packages without privilege element

CRITICAL RULES:

1. UNDERSTAND INTENT, NOT JUST KEYWORDS
   - "eSIM" user wants SIM services → Travel SIM = related (score 2), Esport gaming = unrelated (score 0)
   - "iPhone" user wants device → Mobile package = unrelated (score 0)

2. BRAND SEARCHES ARE STRICT
   - Searching "Netflix" means they want Netflix, not TrueID
   - Searching "Starbucks" means they want Starbucks, not Cafe Amazon
   - Only score 2+ if the exact brand appears in result

3. DON'T BE FOOLED BY SIMILAR WORDS
   - "eSIM" ≠ "Esport" (completely different)
   - "iPhone" ≠ "โทรศัพท์มือถือ service" (device vs service)

4. THINK: "WOULD THIS RESULT MAKE THE USER HAPPY?"
   - If user searched X and got this result, would they be satisfied?
   - If no → score 0 or 1

EVALUATION INPUT:

Search Keyword: "${keyword}"
Result Title: "${title or ''}"
Result Description: "${description or ''}"

Now evaluate: What is the user trying to accomplish? Does this result help them?

Respond with ONLY one number: 0, 1, 2, or 3"""


def score_single_result(keyword: str, title: str, description: str,
                        timeout: int = GEMINI_TIMEOUT) -> tuple:
    """
    Score a single result using Gemini HTTP API and the `result_scoring_prompt`.

    Returns a tuple of (score or None, raw_response_text).
    """
    try:
        prompt_text = result_scoring_prompt(keyword, title or '', description or '')
        response_text = call_gemini_api(prompt_text, timeout=timeout)
        
        if not response_text:
            return None, "API call failed"
        
        m = re.search(r"\b([0-3])\b", response_text)
        if m:
            return int(m.group(1)), response_text
        else:
            return None, response_text

    except Exception as e:
        return None, str(e)


def test_gemini_prompt_tests(enable_gemini: bool = True, delay: float = 1.0):
    """
    Run a suite of small unit tests against the Gemini prompt to validate scoring.

    If `enable_gemini` is False or Gemini fails, tests will report the raw prompt/response.
    """
    tests = [
        # eSIM tests
        {"keyword": 'eSIM', "title": 'GO Travel SIM | ซิมเน็ตต่างประเทศที่ดีที่สุด', "desc": 'มั่นใจ! 5G บนเครือข่ายพันธิมิตร อันดับ 1 ทั่วโลก', "expected": 2},
        {"keyword": 'eSIM', "title": 'แพ็กเกจโรมมิ่ง GO Travel', "desc": 'แพ็กเกจโรมมิ่ง', "expected": 2},
        {"keyword": 'eSIM', "title": 'EsportMax เล่น 4เกมไม่อั้น', "desc": 'เน็ต 15GB 319 บาท', "expected": 0},
        {"keyword": 'eSIM', "title": 'บริการ eSIM', "desc": 'เปิดใช้งาน eSIM ง่ายๆ', "expected": 3},

        # iPhone tests
        {"keyword": 'iPhone', "title": 'ทรูมูฟเอชประหยัด 15บ.', "desc": 'เพลินกับ Social ทุกแอป', "expected": 0},
        {"keyword": 'iPhone', "title": 'หมวกบักเก็ต So Sweet', "desc": 'ลงทะเบียนรับ หมวกบักเก็ต', "expected": 0},
        {"keyword": 'iPhone', "title": 'iPhone 16 Pro', "desc": 'ผ่อน 0% นาน 24 เดือน', "expected": 3},

        # Netflix tests
        {"keyword": 'Netflix', "title": 'บริษัท ทรู มูฟ เอช', "desc": 'รับสิทธิ์ดูหนัง ดูซีรีย์ บนแอพลิเคชั่น True ID 12 เดือน', "expected": 0},
        {"keyword": 'Netflix', "title": 'Netflix Premium', "desc": 'ดู Netflix ฟรี 3 เดือน สำหรับลูกค้าทรู', "expected": 3},

        # Giveaway tests
        {"keyword": 'แจก (Giveaway)', "title": 'FreeGifts รางวัลชุก', "desc": 'ร่วมสนุกฟรีทั้งลูกค้าทรูและดีแทค', "expected": 3},
        {"keyword": 'แจก (Giveaway)', "title": 'สิทธิพิเศษ', "desc": 'ลูกค้าทรูรับ 5 ทรูพอยท์ จากเกมล่าหาของ', "expected": 2},

        # Jaidee tests
        {"keyword": 'Jaidee Emergency Refill', "title": 'บริการข้อมูลการใช้งาน', "desc": '', "expected": 2},
        {"keyword": 'Jaidee Emergency Refill', "title": 'แพ็กเกจเน็ต 5GB', "desc": 'เน็ตเร็วแรง', "expected": 0},

        # Brand tests
        {"keyword": 'Starbucks', "title": 'Cafe Amazon', "desc": 'รับส่วนลด 50%', "expected": 0},
        {"keyword": 'Starbucks', "title": 'Starbucks', "desc": 'รับส่วนลด 50%', "expected": 3},
        {"keyword": 'SF Cinema', "title": 'Major Cineplex', "desc": 'ดูหนังลด 50%', "expected": 0},
    ]

    print('=== TrueApp Gemini Prompt Tests ===\n')
    passed = 0
    results = []

    for i, t in enumerate(tests):
        if enable_gemini:
            score, raw = score_single_result(t['keyword'], t['title'], t['desc'])
        else:
            # If Gemini disabled, show the prompt we would send
            score = None
            raw = result_scoring_prompt(t['keyword'], t['title'], t['desc'])

        status = '✓ PASS' if score == t['expected'] else '✗ FAIL' if score is not None else '⚠ SKIP'
        if score == t['expected']:
            passed += 1

        print(f"Test {i+1}: {status}")
        print(f"  Keyword: \"{t['keyword']}\"")
        print(f"  Title: \"{t['title']}\"")
        print(f"  Got: {score}, Expected: {t['expected']}")
        print(f"  Raw response / prompt:\n{raw}\n")

        results.append({
            'test_index': i+1,
            'keyword': t['keyword'],
            'title': t['title'],
            'expected': t['expected'],
            'got': score,
            'raw': raw,
            'status': status
        })

        time.sleep(delay)

    total = len(tests)
    print(f"Summary: Passed {passed}/{total} tests")
    return {
        'total': total,
        'passed': passed,
        'results': results
    }