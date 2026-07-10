# ============================================================
# Logo Reward Function
# ============================================================
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional

WEIGHTS = {
    "validity": 0.20,
    "bounds": 0.10,
    "element_count": 0.10,
    "palette": 0.10,
    "no_forbidden": 0.10,
    "no_degenerate": 0.10,
    "prompt_coverage": 0.15,
    "structure": 0.15,
}

ALLOWED_TAGS = {'svg','defs','g','path','circle','rect','ellipse',
    'polygon','polyline','line','linearGradient','radialGradient',
    'stop','clipPath','filter','feGaussianBlur','feOffset','feMerge',
    'feMergeNode','feBlend','feColorMatrix','feComposite','mask','use'}

def extract_svg(content):
    m = re.search(r'(<svg\b[^>]*>.*?</svg>)', content, re.DOTALL)
    return m.group(1) if m else None

def parse_svg(svg):
    try:
        return ET.fromstring(svg), ''
    except ET.ParseError as e:
        return None, f'XML parse error: {e}'

def extract_colors(svg):
    return set(re.findall(r'#[0-9a-fA-F]{6}', svg))

def extract_tags(svg):
    return re.findall(r'</?(\w+)', svg)

def is_repetitive(svg, thr=0.85):
    eles = re.findall(r'<(?:path|circle|rect|ellipse|polygon|line)[^>]*?>', svg)
    if len(eles) < 3: return False
    return len(set(eles)) / len(eles) < (1 - thr)

def extract_keywords(prompt):
    stop = {'a','an','the','is','are','has','have','been','was',
        'were','in','on','at','to','for','of','with','by',
        'and','or','but','its',"it's",'that','this','as',
        'like','from','very','small','large','thin','thick',
        'each','all','also','into','more','than','just',
        'some','other','over','under','set'}
    words = re.findall(r'[a-zA-Z]+', prompt.lower())
    return {w for w in words if len(w) > 2 and w not in stop}

def extract_prompt_colors(prompt):
    return set(re.findall(r'#[0-9a-fA-F]{6}', prompt))

def score_validity(svg):
    if not svg or len(svg.strip()) < 20: return 0.0, 'SVG too short'
    root, err = parse_svg(svg)
    if root is None: return 0.0, f'Invalid XML: {err}'
    ns = '{http://www.w3.org/2000/svg}'
    if root.tag != ns+'svg' and root.tag != 'svg': return 0.2, f'Root: {root.tag}'
    vb = root.get('viewBox','')
    nums = vb.replace(',',' ').split()
    if len(nums)!=4 or nums!=['0','0','256','256']: return 0.5, f'viewBox: {vb}'
    opens = len(re.findall(r'<(\w+)[^>]*>', svg))
    sc = len(re.findall(r'<[^>]+/>', svg))
    cls = len(re.findall(r'</(\w+)>', svg))
    if opens-sc != cls: return 0.6, f'Unbalanced: open={opens} close={cls}'
    return 1.0, 'Valid'

def score_bounds(svg):
    matches = re.findall(r"(cx|cy|r|x|y|rx|ry)\s*=\s*['\"](-?\\.?\d+)[\"']", svg)
    if not matches: return 0.7, 'No position attrs'
    out = sum(1 for _,v in matches if float(v)<-10 or float(v)>266)
    score = max(0.5, 1.0-(out/len(matches))*0.5)
    return score, f'{out}/{len(matches)} out of bounds'

def score_element_count(svg):
    cnt = len(re.findall(r'<(?:path|circle|rect|ellipse|polygon|polyline|line)\b', svg))
    if cnt < 2:   return 0.2, f'Too few: {cnt}'
    if cnt > 100: return 0.3, f'Too many: {cnt}'
    if cnt <= 50: return 1.0, f'Good: {cnt}'
    return 0.8, f'Many: {cnt}'

def score_palette(svg):
    cnt = len(extract_colors(svg))
    if cnt == 0:     return 0.3, 'No hex colors'
    if cnt <= 2:     return 0.5, f'Very limited: {cnt}'
    if cnt <= 15:    return 1.0, f'Good palette: {cnt}'
    if cnt <= 25:    return 0.7, f'Large palette: {cnt}'
    return 0.4, f'Too many: {cnt}'

def score_no_forbidden(svg):
    issues = []
    if '<image' in svg:         issues.append('<image>')
    if '<script' in svg:        issues.append('<script>')
    if '<foreignObject' in svg: issues.append('<foreignObject>')
    if '<iframe' in svg:        issues.append('<iframe>')
    ext = re.findall(r'href\s*=\s*["\']https?://', svg)
    if ext:                     issues.append('external URLs')
    if issues: return 0.0, '; '.join(issues)
    return 1.0, 'Clean'

def score_no_degenerate(svg):
    issues = []
    if len(svg.strip()) < 100:   issues.append('too short')
    if len(svg.strip()) > 10000: issues.append('too long')
    if is_repetitive(svg):       issues.append('repetitive')
    if issues: return max(0.0, 1.0-0.3*len(issues)), '; '.join(issues)
    return 1.0, 'Good'

def score_structure(svg):
    issues = []
    tags = extract_tags(svg)
    unknown = set(tags) - ALLOWED_TAGS
    unknown = {t.split('}')[-1] for t in unknown} - ALLOWED_TAGS - {''}
    if unknown:            issues.append(f'unknown tags: {unknown}')
    if 'xmlns' not in svg: issues.append('missing xmlns')
    go = svg.count('<g>') + svg.count('<g ')
    gc = svg.count('</g>')
    if go != gc:           issues.append(f'unbalanced <g>: {go}/{gc}')
    if issues: return max(0.0, 1.0-0.25*len(issues)), '; '.join(issues)
    return 1.0, 'Good structure'

def score_prompt_coverage(svg, prompt):
    keywords = extract_keywords(prompt)
    if not keywords: return 0.5, 'No keywords'
    pcols = extract_prompt_colors(prompt)
    scols = extract_colors(svg)
    svg_lower = svg.lower()
    matched = {kw for kw in keywords if kw in svg_lower}
    kw_cov = len(matched)/len(keywords)
    col_ratio = len(pcols & scols)/max(len(pcols),1)
    return 0.7*kw_cov+0.3*col_ratio, f'KW:{len(matched)}/{len(keywords)} Col:{len(pcols&scols)}/{len(pcols)}'

def compute_reward(svg, prompt='', verbose=False):
    results, details = {}, {}
    ext = extract_svg(svg)
    if ext: svg = ext
    results['validity'], details['validity'] = score_validity(svg)
    results['bounds'], details['bounds'] = score_bounds(svg)
    results['element_count'], details['element_count'] = score_element_count(svg)
    results['palette'], details['palette'] = score_palette(svg)
    results['no_forbidden'], details['no_forbidden'] = score_no_forbidden(svg)
    results['no_degenerate'], details['no_degenerate'] = score_no_degenerate(svg)
    results['structure'], details['structure'] = score_structure(svg)
    if prompt:
        results['prompt_coverage'], details['prompt_coverage'] = score_prompt_coverage(svg, prompt)
    else:
        results['prompt_coverage'] = 0.0
        details['prompt_coverage'] = 'No prompt'
    if verbose:
        for k in results: print(f'  {k:15}: {results[k]:.3f} - {details[k]}')
    total = sum(results[k]*WEIGHTS.get(k,0.1) for k in results)
    return {'total': round(total,4), 'sub_scores': results, 'details': details}