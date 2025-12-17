#!/usr/bin/env python3
"""
Professor Search Engine (Speed Optimized Edition)
使用 BGE-Base 模型 + MPS 加速，解决 M4 上运行缓慢的问题
"""

import os
import sys

# --- 关键配置：设置国内镜像加速 ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 禁用 Tokenizers 并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from openai import OpenAI

# 路径配置
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CRAWLED_DIR = PROCESSED_DIR / "crawled_homepages"
# 向量缓存文件
VECTOR_CACHE_FILE = PROCESSED_DIR / "professor_vectors.npy"

# Qwen AI 配置
QWEN_API_KEY = "sk-a6356e618255431a941a47afeb99e4b1"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class ProfessorSearchEngine:
    """教授搜索引擎 (极速版)"""
    
    def __init__(self, enable_ai_summary: bool = True):
        """初始化搜索引擎"""
        print("🚀 初始化搜索引擎 (极速优化版)...")
        
        # 1. 加载基础数据
        self.countries = self._load_countries()
        self.professors = self._load_professors()
        self.homepage_contents = self._load_homepage_contents()
        
        # 2. 初始化 Embedding 模型 (更换为 Base 模型)
        self._init_embedding_model()
        
        # 3. 构建或加载向量索引
        self._build_vector_index()
        
        # 4. 初始化 Qwen AI 客户端
        self.enable_ai_summary = enable_ai_summary
        if enable_ai_summary:
            try:
                self.qwen_client = OpenAI(
                    api_key=QWEN_API_KEY,
                    base_url=QWEN_BASE_URL
                )
                print(f"✅ Qwen AI 已启用")
            except Exception as e:
                print(f"⚠️  Qwen AI 初始化失败: {e}")
                self.enable_ai_summary = False
                self.qwen_client = None
        else:
            self.qwen_client = None
        
        print(f"✅ 搜索引擎初始化完成！")
        print(f"   - 国家/地区: {len(self.countries)} 个")
        print(f"   - 教授总数: {len(self.professors)} 位")
        print(f"   - 有主页内容: {len(self.homepage_contents)} 位")
        print()

    def _init_embedding_model(self):
        """初始化模型并配置硬件加速"""
        # 检测硬件设备
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("🚀 检测到 Apple Silicon (M4)，已启用 MPS GPU 加速！")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("🚀 检测到 NVIDIA GPU，已启用 CUDA 加速！")
        else:
            self.device = "cpu"
            print("⚠️ 未检测到 GPU，将使用 CPU 运行 (速度较慢)")

        # 使用 Base 模型，速度比 M3 快 5-10 倍，效果依然优秀
        model_name = 'BAAI/bge-base-zh-v1.5'
        print(f"📥 正在加载轻量级模型 {model_name}...")
        
        try:
            self.embedding_model = SentenceTransformer(
                model_name, 
                device=self.device
            )
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise e

    def _load_countries(self) -> Dict:
        """加载国家和地区信息"""
        countries_file = RAW_DIR / "countries.csv"
        countries = {}
            
        with open(countries_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    alpha_2 = row['alpha_2'].lower()
                    countries[alpha_2] = {
                        'name': row['name'],
                        'region': row['region'],
                        'sub_region': row['sub_region'],
                        'alpha_2': alpha_2
                    }
                except KeyError as e:
                    print(f"⚠️  跳过行（缺少字段 {e}）: {row}")
                    continue
        return countries
    
    def _load_professors(self) -> List[Dict]:
        """加载教授基本信息"""
        professors_file = PROCESSED_DIR / "professors.json"
        with open(professors_file, 'r', encoding='utf-8') as f:
            return json.load(f)
        
        return professors
    
    def _load_homepage_contents(self) -> Dict[str, Dict]:
        """加载教授主页内容"""
        contents = {}
        if not CRAWLED_DIR.exists():
            print(f"⚠️  警告: 爬取目录不存在: {CRAWLED_DIR}")
            return contents
        
        for json_file in CRAWLED_DIR.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('status') == 'success' and data.get('content'):
                        name = data.get('name', json_file.stem.replace('_', ' '))
                        contents[name] = {
                            'content': data['content'],
                            'content_length': data.get('content_length', len(data['content'])),
                            'homepage': data.get('homepage', ''),
                            'method': data.get('method', 'unknown')
                        }
            except Exception:
                print(f"⚠️  读取文件失败 {json_file.name}: {e}")
                continue
        return contents
    
    def _build_vector_index(self):
        """构建语义向量索引"""
        print("📊 构建语义索引...")
        
        self.indexed_professors = []
        documents = []
        
        # 1. 准备文档数据
        for i, prof in enumerate(self.professors):
            name = prof['name']
            if name in self.homepage_contents:
                prof_data = {
                    **prof,
                    'homepage_data': self.homepage_contents[name],
                    'vector_index': len(self.indexed_professors)
                }
                self.indexed_professors.append(prof_data)
                
                # 组合文本：Name + Institution + Content
                # 这里可以适当放宽到 1000 字符，因为 Base 模型很快
                content = self.homepage_contents[name]['content']
                doc_text = f"{prof['name']} {prof['institution']} {content[:1000]}"
                documents.append(doc_text)
        
        if not documents:
            print("⚠️  没有可索引的文档")
            return

        # 2. 尝试加载缓存
        if VECTOR_CACHE_FILE.exists():
            print(f"   - 发现缓存文件，正在加载...")
            try:
                self.doc_embeddings = np.load(VECTOR_CACHE_FILE)
                # 校验数量是否一致
                if len(self.doc_embeddings) == len(documents):
                    # 简单校验维度 (Base模型是768维，M3是1024维)
                    if self.doc_embeddings.shape[1] == 768:
                        print("   ✅ 向量缓存加载成功")
                        return
                    else:
                        print("   ⚠️ 缓存维度不匹配（可能是旧模型留下的），准备重新计算...")
                else:
                    print(f"   ⚠️ 缓存数量不一致，重新计算...")
            except Exception as e:
                print(f"   ⚠️ 读取缓存失败: {e}")
        
        # 3. 计算向量 (如果没有有效缓存)
        print(f"   - 正在计算 {len(documents)} 位教授的向量...")
        print("   - 使用 bge-base 模型，预计需要 2-3 分钟...")
        
        # [优化重点] 
        # Base 模型比较轻，M4 可以跑 Batch Size 32 甚至 64
        # Max Length 设为 512 足够覆盖简介
        self.embedding_model.max_seq_length = 512
        
        self.doc_embeddings = self.embedding_model.encode(
            documents,
            batch_size=32,          # M4 跑 Base 模型可以用 32
            show_progress_bar=True,
            normalize_embeddings=True, 
            device=self.device
        )
        
        # 4. 保存缓存
        print(f"   - 正在保存向量缓存...")
        np.save(VECTOR_CACHE_FILE, self.doc_embeddings)
        print("   ✅ 索引构建完成")

    def get_filter_options(self) -> Dict:
        """获取过滤选项"""
        regions = set()
        sub_regions = set()
        countries_list = []
        
        for alpha_2, info in self.countries.items():
            if info['region']:
                regions.add(info['region'])
            if info['sub_region']:
                sub_regions.add(info['sub_region'])
            countries_list.append({
                'name': info['name'],
                'alpha_2': alpha_2,
                'region': info['region'],
                'sub_region': info['sub_region']
            })
        
        return {
            'regions': sorted(list(regions)),
            'sub_regions': sorted(list(sub_regions)),
            'countries': sorted(countries_list, key=lambda x: x['name'])
        }

    def filter_by_location(self, regions=None, sub_regions=None, countries=None) -> List[Dict]:
        """按地区/国家过滤教授"""
        filtered = []
        regions = [r.strip() for r in (regions or [])]
        sub_regions = [sr.strip() for sr in (sub_regions or [])]
        countries = [c.strip().lower() for c in (countries or [])]
        
        if not regions and not sub_regions and not countries:
            return self.indexed_professors
        
        for prof in self.indexed_professors:
            country_code = prof.get('country', '').lower()
            country_info = self.countries.get(country_code, {})
            
            if not country_info:
                # 如果找不到国家信息，跳过这个教授
                continue
            
            country_name = country_info.get('name', '').lower()
            country_region = country_info.get('region', '')
            country_sub_region = country_info.get('sub_region', '')
            
            match = True
            if regions and country_info.get('region') not in regions: match = False
            if match and sub_regions and country_info.get('sub_region') not in sub_regions: match = False
            # 如果指定了 countries，必须匹配
            if countries and match:
                country_match = False
                # 检查国家代码
                if country_code in countries:
                    country_match = True
                # 检查国家名称（完整匹配或部分匹配）
                if not country_match:
                    for c in countries:
                        if c in country_name or country_name in c:
                            country_match = True
                            break
                if not country_match:
                    match = False
            
            if match:
                filtered.append(prof)
        
        return filtered

    def search(self, 
               query: str,
               top_k: int = 20,
               regions: List[str] = None,
               sub_regions: List[str] = None,
               countries: List[str] = None,
               normalize_by_length: bool = True,
               generate_summary: bool = True) -> Dict:
        """执行语义搜索"""
        
        if not query or not query.strip():
            return {'query': query, 'results': [], 'ai_summary': ''}
            
        # 1. 过滤
        filtered_profs = self.filter_by_location(regions, sub_regions, countries)
        if not filtered_profs:
            return {'query': query, 'results': [], 'ai_summary': ''}
            
        target_indices = [p['vector_index'] for p in filtered_profs]
        
        print(f"🔍 语义搜索: '{query}' (范围: {len(target_indices)} 人)")
        
        # 2. Query 向量化
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        
        # 3. 相似度计算
        target_embeddings = self.doc_embeddings[target_indices]
        scores = query_embedding @ target_embeddings.T
        scores = scores[0]
        
        # 4. 排序
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_k_indices, 1):
            score = float(scores[idx])
            prof = filtered_profs[idx]
            
            content = prof['homepage_data']['content']
            # 简单展示前 300 字符
            snippet = self._extract_snippet(content, query)
            
            results.append({
                'rank': rank,
                'name': prof['name'],
                'institution': prof['institution'],
                'country': prof.get('countryName', prof.get('country', '')),
                'region': prof.get('region', ''),
                'homepage': prof.get('homepage', ''),
                'scholarId': prof.get('scholarId', ''),
                'similarity_score': score,
                'content_length': prof['homepage_data']['content_length'],
                'snippet': snippet,
                'research_areas': prof.get('researchAreas', [])
            })
            
        # 5. AI 总结
        ai_summary = ""
        if generate_summary and self.enable_ai_summary:
            ai_summary = self.generate_ai_summary(query, results)
            
        return {
            'query': query,
            'results': results,
            'ai_summary': ai_summary
        }
    def _extract_snippet(self, content: str, query: str, context_length: int = 200) -> str:
        """提取包含查询关键词的文本片段"""
        # 清理文本
        content = ' '.join(content.split())
        
        # 查找关键词位置
        query_lower = query.lower()
        content_lower = content.lower()
        
        # 尝试找到查询词的位置
        pos = content_lower.find(query_lower)
        
        if pos == -1:
            # 如果找不到完整匹配，尝试找第一个词
            words = query_lower.split()
            for word in words:
                pos = content_lower.find(word)
                if pos != -1:
                    break
        
        if pos == -1:
            # 如果还是找不到，返回开头
            return content[:context_length] + "..."
        
        # 提取上下文
        start = max(0, pos - context_length // 2)
        end = min(len(content), pos + len(query) + context_length // 2)
        
        snippet = content[start:end]
        
        # 添加省略号
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        
        return snippet
    def generate_ai_summary(self, query: str, results: List[Dict]) -> str:
        """调用 Qwen 生成总结"""
        if not self.enable_ai_summary or not self.qwen_client:
            return ""
        if not results:
            return "没有搜索结果可以总结"

        num_results = min(len(results), 50) # 减少给 LLM 的上下文量，提高速度
        professors_info = []
        for i, result in enumerate(results[:num_results], 1):
            info = f"{i}. {result['name']} ({result['institution']})\n"
            info += f"   研究内容: {result['snippet'][:150]}\n"
            professors_info.append(info)
        
        professors_text = "\n".join(professors_info)
        prompt = f"""你是一位学术研究助手。用户搜索了关键词"{query}"，找到了以下{num_results}位教授。

请根据这些教授的信息，生成一段简洁的总结（200-300字），包括：
1. 这些教授的主要研究方向和共同点
2. 他们所在的主要机构和地区分布
3. 研究领域的特点和趋势
4. 对用户寻找合适导师的建议

教授信息：
{professors_text}

请用中文回答，语言简洁专业。"""

        try:
            print("\n🤖 正在生成 AI 总结...")
            response = self.qwen_client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "你是一位专业的学术研究助手，擅长分析教授的研究方向和提供学术建议。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"AI 总结生成失败: {e}")
            return "AI 总结生成失败。"

    def print_results(self, search_result: Dict):
        """打印结果"""
        results = search_result.get('results', [])
        ai_summary = search_result.get('ai_summary', '')
        if not results:
            print("❌ 没有找到结果")
            return
            
        # 显示 AI 总结
        if show_ai_summary and ai_summary:
            print(f"\n{'='*80}")
            print(f"🤖 AI 总结")
            print(f"{'='*80}")
            print(ai_summary)
            print()
        # 显示搜索结果
        print(f"\n{'='*80}")
        print(f"找到 {len(results)} 个结果")
        print(f"{'='*80}\n")
        
        for res in results[:5]:
            print(f"#{res['rank']} {res['name']} ({res['similarity_score']:.4f})")
            print(f"   🏫 {res['institution']} ({res['country']})")
            #print(f"   📊 相似度: {res['similarity_score']:.4f}")
            print(f"   📏 内容长度: {res['content_length']} 字符")
            if res['homepage']:
                print(f"   🔗 主页: {res['homepage']}")
            
            if res['research_areas']:
                print(f"   🔬 研究领域: {', '.join(res['research_areas'][:5])}")
            
            if show_snippet and res['snippet']:
                print(f"   📝 片段: {res['snippet'][:300]}")
            
            print()

def main():
    """测试入口"""
    engine = ProfessorSearchEngine()
    print("\n🔍 测试: 搜索 '信息检索'...")
    # 示例1: 无过滤的搜索（带 AI 总结）
    print("\n" + "="*80)
    print("示例 1: 搜索 'reinforcement learning, sequential decision making' (无过滤, top 20)")
    print("="*80)
    res = engine.search("reinforcement learning, sequential decision making", top_k=20)
    engine.print_results(res)

    # 示例2: 按地区过滤（带 AI 总结）
    print("\n" + "="*80)
    print("示例 2: 搜索 'machine learning' (仅亚洲, top 20)")
    print("="*80)
    res = engine.search("machine learning", top_k=20, regions=["Asia"])
    engine.print_results(res)

    # 示例3: 按国家过滤（带 AI 总结）
    print("\n" + "="*80)
    print("示例 3: 搜索 'recommender system and large language models' (仅美国, top 20)")
    print("="*80)
    res = engine.search(
        "recommender system and large language models", top_k=20, countries=["United States", "us"]
    )
    engine.print_results(res)

if __name__ == "__main__":
    main()