#!/usr/bin/env python3
"""
Professor Search Engine
支持区域/国家过滤和基于内容相似度的搜索
"""

import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
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

# Qwen AI 配置
QWEN_API_KEY = "sk-a6356e618255431a941a47afeb99e4b1"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class ProfessorSearchEngine:
    """教授搜索引擎"""
    
    def __init__(self, enable_ai_summary: bool = True):
        """初始化搜索引擎"""
        print("🚀 初始化搜索引擎...")
        
        # 加载数据
        self.countries = self._load_countries()
        self.professors = self._load_professors()
        self.homepage_contents = self._load_homepage_contents()
        
        # 构建索引
        self._build_search_index()
        
        # 初始化 Qwen AI 客户端
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
    
    def _load_countries(self) -> Dict:
        """加载国家和地区信息"""
        countries_file = RAW_DIR / "countries.csv"
        countries = {}
        
        with open(countries_file, 'r', encoding='utf-8-sig') as f:  # 处理BOM
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
            professors = json.load(f)
        
        return professors
    
    def _load_homepage_contents(self) -> Dict[str, Dict]:
        """加载教授主页内容"""
        contents = {}
        
        if not CRAWLED_DIR.exists():
            print(f"⚠️  警告: 爬取目录不存在: {CRAWLED_DIR}")
            return contents
        
        # 遍历所有JSON文件
        for json_file in CRAWLED_DIR.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 只保留成功爬取的内容
                    if data.get('status') == 'success' and data.get('content'):
                        name = data.get('name', json_file.stem.replace('_', ' '))
                        contents[name] = {
                            'content': data['content'],
                            'content_length': data.get('content_length', len(data['content'])),
                            'homepage': data.get('homepage', ''),
                            'method': data.get('method', 'unknown')
                        }
            except Exception as e:
                print(f"⚠️  读取文件失败 {json_file.name}: {e}")
                continue
        
        return contents
    
    def _build_search_index(self):
        """构建搜索索引（TF-IDF）"""
        print("📊 构建搜索索引...")
        
        # 准备文档列表
        self.indexed_professors = []
        documents = []
        
        for prof in self.professors:
            name = prof['name']
            if name in self.homepage_contents:
                self.indexed_professors.append({
                    **prof,
                    'homepage_data': self.homepage_contents[name]
                })
                documents.append(self.homepage_contents[name]['content'])
        
        print(f"   - 可搜索的教授: {len(self.indexed_professors)} 位")
        
        # 构建TF-IDF向量
        if documents:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,  # 减少特征数
                stop_words='english',
                ngram_range=(1, 1),  # 只使用1-gram，减少内存
                min_df=3,  # 增加最小文档频率
                max_df=0.7  # 降低最大文档频率
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(documents)
            print(f"   - TF-IDF矩阵: {self.tfidf_matrix.shape}")
            print(f"   - 矩阵大小: ~{self.tfidf_matrix.data.nbytes / 1024 / 1024:.2f} MB")
        else:
            self.vectorizer = None
            self.tfidf_matrix = None
            print("   ⚠️  没有文档可以索引")
    
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
    
    def filter_by_location(self, 
                          regions: List[str] = None,
                          sub_regions: List[str] = None,
                          countries: List[str] = None) -> List[Dict]:
        """
        按地区/国家过滤教授
        
        Args:
            regions: 地区列表（如 ["Asia", "Europe"]）
            sub_regions: 子地区列表（如 ["Southern Asia", "Western Europe"]）
            countries: 国家列表（如 ["United States", "China", "us", "cn"]）
        
        Returns:
            过滤后的教授列表
        """
        filtered = []
        
        # 标准化输入
        regions = [r.strip() for r in (regions or [])]
        sub_regions = [sr.strip() for sr in (sub_regions or [])]
        countries = [c.strip().lower() for c in (countries or [])]
        
        # 如果没有任何过滤条件，返回所有有主页内容的教授
        if not regions and not sub_regions and not countries:
            return self.indexed_professors
        
        for prof in self.indexed_professors:
            country_code = prof.get('country', '').lower()
            
            # 获取国家信息（从 countries.csv，不使用 professors.json 中的 region）
            country_info = self.countries.get(country_code, {})
            if not country_info:
                # 如果找不到国家信息，跳过这个教授
                continue
            
            country_name = country_info.get('name', '').lower()
            country_region = country_info.get('region', '')
            country_sub_region = country_info.get('sub_region', '')
            
            # 检查是否匹配（使用 AND 逻辑：所有指定的条件都必须满足）
            match = True
            
            # 如果指定了 regions，必须匹配
            if regions:
                if country_region not in regions:
                    match = False
            
            # 如果指定了 sub_regions，必须匹配
            if sub_regions and match:
                if country_sub_region not in sub_regions:
                    match = False
            
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
        """
        搜索教授
        
        Args:
            query: 搜索关键词
            top_k: 返回前K个结果（默认20）
            regions: 地区过滤
            sub_regions: 子地区过滤
            countries: 国家过滤
            normalize_by_length: 是否按内容长度归一化
            generate_summary: 是否生成 AI 总结
        
        Returns:
            包含搜索结果和 AI 总结的字典
            {
                'query': str,
                'results': List[Dict],
                'ai_summary': str
            }
        """
        if not query or not query.strip():
            print("⚠️  搜索关键词为空")
            return {'query': query, 'results': [], 'ai_summary': ''}
        
        if self.vectorizer is None or self.tfidf_matrix is None:
            print("⚠️  搜索索引未构建")
            return {'query': query, 'results': [], 'ai_summary': ''}
        
        # 先按地区/国家过滤
        filtered_profs = self.filter_by_location(regions, sub_regions, countries)
        
        if not filtered_profs:
            print("⚠️  没有符合过滤条件的教授")
            return {'query': query, 'results': [], 'ai_summary': ''}
        
        print(f"🔍 搜索: '{query}'")
        print(f"   - 过滤后的教授数: {len(filtered_profs)} 位")
        
        # 获取过滤后教授的索引
        filtered_indices = []
        for prof in filtered_profs:
            try:
                idx = self.indexed_professors.index(prof)
                filtered_indices.append(idx)
            except ValueError:
                continue
        
        if not filtered_indices:
            print("⚠️  没有可搜索的教授")
            return {'query': query, 'results': [], 'ai_summary': ''}
        
        # 将查询转换为TF-IDF向量
        query_vector = self.vectorizer.transform([query])
        
        # 计算余弦相似度
        similarities = cosine_similarity(query_vector, self.tfidf_matrix[filtered_indices]).flatten()
        
        # 如果需要按长度归一化
        if normalize_by_length:
            for i, idx in enumerate(filtered_indices):
                prof = self.indexed_professors[idx]
                content_length = prof['homepage_data']['content_length']
                
                # 归一化因子：使用对数缩放，避免过度惩罚长文档
                length_factor = np.log(1 + content_length) / np.log(1 + 10000)  # 假设10000是平均长度
                length_factor = min(length_factor, 1.5)  # 限制最大影响
                
                similarities[i] = similarities[i] / length_factor
        
        # 获取top-k结果
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for rank, i in enumerate(top_indices, 1):
            idx = filtered_indices[i]
            prof = self.indexed_professors[idx]
            
            # 获取匹配的关键词片段
            content = prof['homepage_data']['content']
            snippet = self._extract_snippet(content, query)
            
            results.append({
                'rank': rank,
                'name': prof['name'],
                'institution': prof['institution'],
                'country': prof.get('countryName', prof.get('country', '')),
                'region': prof.get('region', ''),
                'homepage': prof.get('homepage', ''),
                'scholarId': prof.get('scholarId', ''),
                'similarity_score': float(similarities[i]),
                'content_length': prof['homepage_data']['content_length'],
                'snippet': snippet,
                'research_areas': prof.get('researchAreas', [])
            })
        
        # 生成 AI 总结
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
        """
        使用 Qwen AI 生成搜索结果的总结
        
        Args:
            query: 搜索关键词
            results: 搜索结果列表
        
        Returns:
            AI 生成的总结文本
        """
        if not self.enable_ai_summary or not self.qwen_client:
            return "AI 总结功能未启用"
        
        if not results:
            return "没有搜索结果可以总结"
        
        # 使用所有结果（最多50个）进行总结
        num_results = min(len(results), 50)
        
        # 准备提示词
        professors_info = []
        for i, result in enumerate(results[:num_results], 1):
            info = f"{i}. {result['name']} - {result['institution']} ({result['country']})\n"
            info += f"   相似度: {result['similarity_score']:.4f}\n"
            if result['snippet']:
                info += f"   研究内容: {result['snippet'][:500]}\n"
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
                model="qwen-plus",  # 使用 qwen-plus 模型
                messages=[
                    {"role": "system", "content": "你是一位专业的学术研究助手，擅长分析教授的研究方向和提供学术建议。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            summary = response.choices[0].message.content
            return summary
            
        except Exception as e:
            return f"AI 总结生成失败: {e}"
    
    def print_results(self, search_result: Dict, show_snippet: bool = True, show_ai_summary: bool = True):
        """
        打印搜索结果
        
        Args:
            search_result: 搜索结果字典，包含 'query', 'results', 'ai_summary'
            show_snippet: 是否显示文本片段
            show_ai_summary: 是否显示 AI 总结
        """
        results = search_result.get('results', [])
        ai_summary = search_result.get('ai_summary', '')
        
        if not results:
            print("❌ 没有找到匹配的结果")
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
        
        for result in results:
            print(f"🏆 #{result['rank']} - {result['name']}")
            print(f"   🏫 {result['institution']} ({result['country']})")
            print(f"   📊 相似度: {result['similarity_score']:.4f}")
            print(f"   📏 内容长度: {result['content_length']} 字符")
            
            if result['homepage']:
                print(f"   🔗 主页: {result['homepage']}")
            
            if result['research_areas']:
                print(f"   🔬 研究领域: {', '.join(result['research_areas'][:5])}")
            
            if show_snippet and result['snippet']:
                print(f"   📝 片段: {result['snippet'][:300]}")
            
            print()


def main():
    """主函数 - 示例用法"""
    # 初始化搜索引擎
    engine = ProfessorSearchEngine()

    # 示例1: 无过滤的搜索（带 AI 总结）
    print("\n" + "="*80)
    print("示例 1: 搜索 'reinforcement learning, sequential decision making' (无过滤, top 20)")
    print("="*80)
    results = engine.search("reinforcement learning, sequential decision making", top_k=20)
    engine.print_results(results)

    # 示例2: 按地区过滤（带 AI 总结）
    print("\n" + "="*80)
    print("示例 2: 搜索 'machine learning' (仅亚洲, top 20)")
    print("="*80)
    results = engine.search("machine learning", top_k=20, regions=["Asia"])
    engine.print_results(results)

    # 示例3: 按国家过滤（带 AI 总结）
    print("\n" + "="*80)
    print("示例 3: 搜索 'recommender system and large language models' (仅美国, top 20)")
    print("="*80)
    results = engine.search(
        "recommender system and large language models", top_k=20, countries=["United States", "us"]
    )
    engine.print_results(results)


if __name__ == "__main__":
    main()
