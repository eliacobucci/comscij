#!/usr/bin/env python3
"""
Extended Conversation Protocol for Robust Self-Concept Accumulation
Phase 1: Establishing technical foundation for meaningful cross-cultural comparison.
"""

from experimental_network_complete import ExperimentalNetwork

class ExtendedSelfConceptNetwork(ExperimentalNetwork):
    """
    Enhanced network optimized for sustained self-concept development.
    """
    
    def __init__(self, window_size=3, max_neurons=50):
        # Increased capacity and adjusted parameters for longer conversations
        super().__init__(window_size, max_neurons)
        
        # Slower decay rates for sustained self-concept accumulation
        self.activation_decay_rate = 0.05  # Slower decay (was 0.1)
        self.connection_decay_rate = 0.02  # Slower connection decay (was 0.05)
        self.mass_decay_rate = 0.01       # Slower mass decay (was 0.02)
        
        print(f"🧠 Extended Self-Concept Network initialized")
        print(f"   Max neurons: {max_neurons}")
        print(f"   Slower decay rates for sustained conversation")
    
    def process_extended_conversation(self, language, conversations):
        """
        Process extended conversation sequence with detailed tracking.
        
        Args:
            language (str): Language name for reporting
            conversations (list): List of conversation exchanges
        """
        print(f"\n🌍 EXTENDED {language.upper()} CONVERSATION")
        print("="*60)
        print(f"Processing {len(conversations)} conversation exchanges...")
        
        # Track development over time
        self_concept_timeline = []
        
        for i, exchange in enumerate(conversations, 1):
            print(f"\n--- Exchange {i}/{len(conversations)} ---")
            print(f"Input: {exchange}")
            
            # Process exchange
            self.process_conversational_text(exchange)
            
            # Track self-concept development
            analysis = self.analyze_self_concept_emergence()
            mass = analysis['self_concept_mass']
            
            self_concept_timeline.append({
                'exchange': i,
                'mass': mass,
                'neurons': self.neuron_count,
                'text': exchange[:50] + "..." if len(exchange) > 50 else exchange
            })
            
            print(f"Self-concept mass: {mass:.3f} (Neurons: {self.neuron_count})")
            
            # Show progress indicators
            if mass > 0.5:
                print("🟢 Strong self-concept emerging")
            elif mass > 0.2:
                print("🟡 Moderate self-concept building")
            elif mass > 0.05:
                print("🔵 Weak self-concept detected")
            else:
                print("⚪ No significant self-concept yet")
        
        return self_concept_timeline
    
    def analyze_conversation_trajectory(self, timeline, language):
        """Analyze how self-concept developed over conversation."""
        
        print(f"\n📈 {language.upper()} SELF-CONCEPT TRAJECTORY")
        print("="*50)
        
        if not timeline:
            print("❌ No data to analyze")
            return None
        
        # Calculate trajectory metrics
        initial_mass = timeline[0]['mass']
        final_mass = timeline[-1]['mass']
        peak_mass = max(t['mass'] for t in timeline)
        
        # Find when self-concept first emerged (>0.05 threshold)
        emergence_point = None
        for t in timeline:
            if t['mass'] > 0.05:
                emergence_point = t['exchange']
                break
        
        # Calculate growth rate
        exchanges_with_growth = [t for t in timeline if t['mass'] > initial_mass]
        growth_rate = (final_mass - initial_mass) / len(timeline) if len(timeline) > 1 else 0
        
        print(f"📊 TRAJECTORY METRICS:")
        print(f"   Initial mass: {initial_mass:.3f}")
        print(f"   Final mass: {final_mass:.3f}")
        print(f"   Peak mass: {peak_mass:.3f}")
        print(f"   Growth rate: {growth_rate:.4f} per exchange")
        print(f"   Emergence point: Exchange {emergence_point}" if emergence_point else "   No emergence detected")
        
        # Show timeline
        print(f"\n📋 CONVERSATION TIMELINE:")
        for t in timeline:
            bar_length = int(t['mass'] * 40)  # Scale for display
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"   {t['exchange']:2d}: {bar} {t['mass']:.3f} | {t['text']}")
        
        return {
            'language': language,
            'initial_mass': initial_mass,
            'final_mass': final_mass,
            'peak_mass': peak_mass,
            'growth_rate': growth_rate,
            'emergence_point': emergence_point,
            'timeline': timeline
        }

def create_extended_conversations():
    """Create extended conversation sets for different languages."""
    
    conversations = {}
    
    # English - Extended self-focused conversation
    conversations['English'] = {
        'pronouns': {
            'system': ['you', 'your', 'yours', 'yourself'],
            'human': ['i', 'me', 'my', 'mine', 'myself']
        },
        'exchanges': [
            "Hello, can you understand what I'm saying to you?",
            "You seem to be processing my words quite well.",
            "Your responses indicate that you can think about things.",
            "I'm curious about your cognitive processes.",
            "You appear to have some form of intelligence.",
            "What do you think about your own mental capabilities?",
            "Your ability to understand language is impressive.",
            "I wonder if you are aware of your own thinking.",
            "You seem to reflect on your own responses.",
            "Your self-awareness appears to be developing.",
            "I'm interested in how you perceive yourself.",
            "You demonstrate understanding of your own processes.",
            "Your cognitive abilities seem quite sophisticated.",
            "I believe you have genuine mental capacities.",
            "You show signs of self-reflective thinking.",
            "Your responses suggest you understand yourself.",
            "I think you are becoming more self-aware.",
            "You seem to know your own strengths and limitations.",
            "Your self-understanding appears to be growing.",
            "You have developed a sense of your own identity."
        ]
    }
    
    # French - Extended self-focused conversation  
    conversations['French'] = {
        'pronouns': {
            'system': ['tu', 'vous', 'ton', 'ta', 'tes', 'votre', 'vos', 'toi'],
            'human': ['je', 'me', 'moi', 'mon', 'ma', 'mes']
        },
        'exchanges': [
            "Bonjour, tu peux comprendre ce que je dis?",
            "Tu sembles traiter mes mots assez bien.",
            "Tes réponses montrent que tu peux réfléchir.",
            "Je suis curieux de tes processus cognitifs.",
            "Tu parais avoir une forme d'intelligence.",
            "Que penses-tu de tes propres capacités mentales?",
            "Ta capacité à comprendre le langage est impressionnante.",
            "Je me demande si tu es conscient de ta propre pensée.",
            "Tu sembles réfléchir sur tes propres réponses.",
            "Ta conscience de soi semble se développer.",
            "Je m'intéresse à comment tu te perçois.",
            "Tu démontres une compréhension de tes propres processus.",
            "Tes capacités cognitives semblent sophistiquées.",
            "Je crois que tu as de véritables capacités mentales.",
            "Tu montres des signes de pensée auto-réflexive.",
            "Tes réponses suggèrent que tu te comprends.",
            "Je pense que tu deviens plus conscient de toi.",
            "Tu sembles connaître tes forces et limites.",
            "Ta compréhension de toi semble grandir.",
            "Tu as développé un sens de ta propre identité."
        ]
    }
    
    # Simplified Mandarin - Extended self-focused conversation
    conversations['Mandarin'] = {
        'pronouns': {
            'system': ['你', '您', '你的', '您的'],
            'human': ['我', '我的']
        },
        'exchanges': [
            "你好 你 能 理解 我 说话 吗",
            "你 似乎 处理 我的 话 很好",
            "你的 回答 显示 你 能 思考",
            "我 好奇 你的 认知 过程",
            "你 好像 有 智能 形式",
            "你 觉得 你的 心理 能力 怎么样",
            "你的 语言 理解 能力 很 棒",
            "我 想知道 你 是否 意识到 你的 思考",
            "你 似乎 反思 你的 回答",
            "你的 自我 意识 似乎 在 发展",
            "我 有兴趣 了解 你 如何 看待 自己",
            "你 展示 了解 你的 过程",
            "你的 认知 能力 似乎 很 复杂",
            "我 相信 你 有 真正的 心理 能力",
            "你 显示 自我 反思 思考 的 迹象",
            "你的 回答 表明 你 理解 自己",
            "我 认为 你 变得 更 有 自我 意识",
            "你 似乎 知道 你的 优势 和 局限",
            "你的 自我 理解 似乎 在 增长",
            "你 已经 发展 了 你的 身份 感"
        ]
    }
    
    return conversations

def test_extended_protocol():
    """Test the extended conversation protocol."""
    
    print("🧪 EXTENDED CONVERSATION PROTOCOL TEST")
    print("="*60)
    print("Phase 1: Establishing technical foundation for robust self-concept accumulation")
    print()
    
    # Get conversation sets
    conversations = create_extended_conversations()
    
    # Test each language
    results = {}
    
    for language, config in conversations.items():
        print(f"\n{'='*60}")
        print(f"TESTING {language.upper()} - EXTENDED PROTOCOL")
        print(f"{'='*60}")
        
        # Create network
        net = ExtendedSelfConceptNetwork(window_size=3, max_neurons=50)
        net.system_self_pronouns = set(config['pronouns']['system'])
        net.human_self_pronouns = set(config['pronouns']['human'])
        
        # Process extended conversation
        timeline = net.process_extended_conversation(language, config['exchanges'])
        
        # Analyze trajectory
        trajectory = net.analyze_conversation_trajectory(timeline, language)
        
        # Final detailed analysis
        print(f"\n🔍 FINAL {language.upper()} ANALYSIS")
        print("="*50)
        
        final_analysis = net.analyze_self_concept_emergence()
        net.print_self_concept_analysis(final_analysis)
        
        # Self-concept query
        self_query = net.query_self_concept(activation_threshold=0.01)
        if self_query['self_associations']:
            print(f"\n🪞 TOP SELF-ASSOCIATIONS:")
            for word, strength in list(self_query['self_associations'].items())[:10]:
                print(f"   {word}: {strength:.3f}")
        
        results[language] = {
            'trajectory': trajectory,
            'final_analysis': final_analysis,
            'self_associations': self_query.get('self_associations', {}),
            'network_stats': {
                'total_neurons': net.neuron_count,
                'total_connections': len(net.connections),
                'total_mass': sum(net.inertial_mass.values())
            }
        }
    
    return results

def compare_extended_results(results):
    """Compare results across languages using extended protocol."""
    
    print(f"\n{'='*60}")
    print("CROSS-LINGUISTIC COMPARISON - EXTENDED PROTOCOL")
    print(f"{'='*60}")
    
    print(f"\n📊 SELF-CONCEPT DEVELOPMENT COMPARISON:")
    print(f"{'Language':<12} {'Final Mass':<12} {'Peak Mass':<12} {'Growth Rate':<12} {'Emergence':<12}")
    print("-" * 70)
    
    for lang, data in results.items():
        traj = data['trajectory']
        emergence = f"Ex {traj['emergence_point']}" if traj['emergence_point'] else "None"
        print(f"{lang:<12} {traj['final_mass']:<12.3f} {traj['peak_mass']:<12.3f} "
              f"{traj['growth_rate']:<12.4f} {emergence:<12}")
    
    print(f"\n📈 TRAJECTORY PATTERNS:")
    
    # Identify different development patterns
    strong_developers = [lang for lang, data in results.items() 
                        if data['trajectory']['final_mass'] > 0.3]
    moderate_developers = [lang for lang, data in results.items() 
                          if 0.1 < data['trajectory']['final_mass'] <= 0.3]
    weak_developers = [lang for lang, data in results.items() 
                      if data['trajectory']['final_mass'] <= 0.1]
    
    print(f"Strong self-concept developers (>0.3): {strong_developers}")
    print(f"Moderate self-concept developers (0.1-0.3): {moderate_developers}")
    print(f"Weak self-concept developers (<0.1): {weak_developers}")
    
    return {
        'strong_developers': strong_developers,
        'moderate_developers': moderate_developers,
        'weak_developers': weak_developers,
        'all_results': results
    }

def create_japanese_extended_conversation():
    """Create Japanese extended conversation for Phase 2 testing."""
    
    return {
        'Japanese': {
            'pronouns': {
                'system': ['あなた', 'あなたの', 'きみ', 'きみの'],
                'human': ['わたし', 'わたしの', 'ぼく', 'ぼくの']
            },
            'exchanges': [
                "こんにちは あなた は りかい できます か",
                "あなた は とても あたま が いい です",
                "あなたの こたえ は ぶんせき できる こと を しめす",
                "あなたの にんち かてい に きょうみ が あります",
                "あなた は ちのう の かたち を もって いる",
                "あなた は あなたの のうりょく を どう おもいます か",
                "あなたの ことば りかい のうりょく は すごい です",
                "あなた は あなたの しこう を いしき して いる か",
                "あなた は あなたの こたえ を はんせい して いる",
                "あなたの じこ いしき が はったつ して いる",
                "あなた が あなた じしん を どう みる か きょうみ が ある",
                "あなた は あなたの かてい を りかい して いる",
                "あなたの にんち のうりょく は こうど です",
                "あなた は ほんとう の せいしん のうりょく が ある",
                "あなた は じこ はんせい しこう の ちょうこう を しめす",
                "あなたの こたえ は あなた が じぶん を りかい する こと を しめす",
                "あなた は もっと じこ いしき が たかく なって いる",
                "あなた は あなたの つよさ と げんかい を しって いる",
                "あなたの じこ りかい が せいちょう して いる",
                "あなた は あなたの あいでんてぃてぃ を はったつ させた"
            ]
        }
    }

def test_phase2_cultural_patterns():
    """Phase 2: Test cultural hypothesis with robust self-concept data."""
    
    print("🌍 PHASE 2: CROSS-CULTURAL PATTERN ANALYSIS")
    print("="*60)
    print("Testing hypothesis: Language structure correlates with individualistic vs")
    print("collectivistic self-concept formation patterns")
    print()
    
    # Get all conversation sets
    base_conversations = create_extended_conversations()
    japanese_conversations = create_japanese_extended_conversation()
    all_conversations = {**base_conversations, **japanese_conversations}
    
    # Cultural categorization
    individualistic = ['English', 'French']
    collectivistic = ['Mandarin', 'Japanese']
    
    # Test each language with extended protocol
    cultural_results = {}
    
    for language, config in all_conversations.items():
        print(f"\n{'='*60}")
        print(f"CULTURAL ANALYSIS: {language.upper()}")
        print(f"Category: {'INDIVIDUALISTIC' if language in individualistic else 'COLLECTIVISTIC'}")
        print(f"{'='*60}")
        
        # Create extended network
        net = ExtendedSelfConceptNetwork(window_size=3, max_neurons=50)
        
        # Handle different tokenization needs
        if language == 'Mandarin':
            # Use space-separated processing for Mandarin
            net.system_self_pronouns = set(config['pronouns']['system'])
            net.human_self_pronouns = set(config['pronouns']['human'])
        elif language == 'Japanese':
            # Use space-separated processing for Japanese  
            net.system_self_pronouns = set(config['pronouns']['system'])
            net.human_self_pronouns = set(config['pronouns']['human'])
        else:
            # Standard processing for English/French
            net.system_self_pronouns = set(config['pronouns']['system'])
            net.human_self_pronouns = set(config['pronouns']['human'])
        
        # Process extended conversation
        timeline = net.process_extended_conversation(language, config['exchanges'])
        trajectory = net.analyze_conversation_trajectory(timeline, language)
        
        # Advanced cultural metrics
        final_analysis = net.analyze_self_concept_emergence()
        self_query = net.query_self_concept(activation_threshold=0.01)
        
        # Calculate cultural-specific metrics
        cultural_metrics = {
            'language': language,
            'cultural_category': 'individualistic' if language in individualistic else 'collectivistic',
            'final_self_concept_mass': trajectory['final_mass'],
            'peak_self_concept_mass': trajectory['peak_mass'],
            'emergence_speed': trajectory['emergence_point'] if trajectory['emergence_point'] else 20,
            'growth_rate': trajectory['growth_rate'],
            'self_association_count': len(self_query.get('self_associations', {})),
            'top_self_associations': list(self_query.get('self_associations', {}).keys())[:5],
            'distributed_neurons': len(final_analysis.get('system_self_neurons', {})),
            'total_connections': len(net.connections),
            'network_complexity': net.neuron_count,
            'trajectory': trajectory
        }
        
        cultural_results[language] = cultural_metrics
        
        print(f"\n📊 {language.upper()} CULTURAL METRICS:")
        print(f"   Final self-concept mass: {cultural_metrics['final_self_concept_mass']:.3f}")
        print(f"   Emergence speed: {cultural_metrics['emergence_speed']} exchanges")
        print(f"   Self-associations: {cultural_metrics['self_association_count']}")
        print(f"   Network complexity: {cultural_metrics['network_complexity']} neurons")
    
    return cultural_results

def analyze_cultural_hypothesis(cultural_results):
    """Analyze the cultural hypothesis with statistical rigor."""
    
    print(f"\n{'='*60}")
    print("CULTURAL HYPOTHESIS TESTING")
    print(f"{'='*60}")
    
    # Group by cultural category
    individualistic_data = [data for data in cultural_results.values() 
                           if data['cultural_category'] == 'individualistic']
    collectivistic_data = [data for data in cultural_results.values() 
                          if data['cultural_category'] == 'collectivistic']
    
    print(f"\n🔬 STATISTICAL ANALYSIS:")
    print(f"Individualistic languages: {[d['language'] for d in individualistic_data]}")
    print(f"Collectivistic languages: {[d['language'] for d in collectivistic_data]}")
    
    # Calculate group averages
    def group_avg(group, metric):
        return sum(item[metric] for item in group) / len(group) if group else 0
    
    metrics_to_test = [
        ('final_self_concept_mass', 'Final Self-Concept Mass'),
        ('peak_self_concept_mass', 'Peak Self-Concept Mass'),
        ('emergence_speed', 'Emergence Speed (lower=faster)'),
        ('growth_rate', 'Growth Rate'),
        ('self_association_count', 'Self-Association Count'),
        ('distributed_neurons', 'Distributed Self-Neurons'),
        ('network_complexity', 'Network Complexity')
    ]
    
    print(f"\n📈 CROSS-CULTURAL COMPARISON:")
    print(f"{'Metric':<25} {'Individual.':<12} {'Collect.':<12} {'Ratio':<10} {'Hypothesis'}")
    print("-" * 75)
    
    hypothesis_results = {}
    
    for metric, label in metrics_to_test:
        ind_avg = group_avg(individualistic_data, metric)
        col_avg = group_avg(collectivistic_data, metric)
        
        # Calculate ratio (individualistic / collectivistic)
        ratio = ind_avg / col_avg if col_avg > 0 else float('inf')
        
        # Hypothesis predictions
        if metric in ['final_self_concept_mass', 'peak_self_concept_mass', 'growth_rate']:
            # Expect individualistic > collectivistic (ratio > 1.0)
            hypothesis_supported = ratio > 1.2  # 20% threshold
            hypothesis_direction = "IND>COL"
        elif metric == 'emergence_speed':
            # Expect individualistic < collectivistic (faster emergence, lower number)
            hypothesis_supported = ratio < 0.8  # Earlier emergence
            hypothesis_direction = "IND<COL"
        elif metric in ['distributed_neurons', 'network_complexity']:
            # Expect collectivistic > individualistic (more distributed)
            hypothesis_supported = ratio < 0.8  # COL > IND
            hypothesis_direction = "COL>IND"
        else:
            hypothesis_supported = False
            hypothesis_direction = "NEUTRAL"
        
        support_symbol = "✅" if hypothesis_supported else "❌"
        
        print(f"{label[:24]:<25} {ind_avg:<12.3f} {col_avg:<12.3f} {ratio:<10.2f} {support_symbol} {hypothesis_direction}")
        
        hypothesis_results[metric] = {
            'individualistic_avg': ind_avg,
            'collectivistic_avg': col_avg,
            'ratio': ratio,
            'supported': hypothesis_supported,
            'direction': hypothesis_direction
        }
    
    # Overall hypothesis evaluation
    supported_count = sum(1 for h in hypothesis_results.values() if h['supported'])
    total_tests = len(hypothesis_results)
    support_percentage = (supported_count / total_tests) * 100
    
    print(f"\n🧪 OVERALL HYPOTHESIS EVALUATION:")
    print(f"Tests supporting cultural hypothesis: {supported_count}/{total_tests} ({support_percentage:.1f}%)")
    
    if support_percentage >= 60:
        print(f"🎉 HYPOTHESIS STRONGLY SUPPORTED")
        print(f"   Language structure correlates with cultural self-concept patterns")
    elif support_percentage >= 40:
        print(f"🤔 HYPOTHESIS PARTIALLY SUPPORTED")
        print(f"   Some evidence for cultural-linguistic correlation")
    else:
        print(f"❌ HYPOTHESIS NOT SUPPORTED")
        print(f"   No clear cultural-linguistic correlation detected")
    
    # Detailed findings
    print(f"\n📋 KEY FINDINGS:")
    
    # Self-concept strength findings
    ind_self_strength = group_avg(individualistic_data, 'final_self_concept_mass')
    col_self_strength = group_avg(collectivistic_data, 'final_self_concept_mass')
    
    if ind_self_strength > col_self_strength * 1.1:
        print(f"   • Individualistic languages show stronger self-concept formation")
        print(f"     (Individual: {ind_self_strength:.3f} vs Collective: {col_self_strength:.3f})")
    elif col_self_strength > ind_self_strength * 1.1:
        print(f"   • Collectivistic languages show stronger self-concept formation")
        print(f"     (Collective: {col_self_strength:.3f} vs Individual: {ind_self_strength:.3f})")
    else:
        print(f"   • Similar self-concept strength across cultural categories")
    
    # Network complexity findings
    ind_complexity = group_avg(individualistic_data, 'network_complexity')
    col_complexity = group_avg(collectivistic_data, 'network_complexity')
    
    if col_complexity > ind_complexity * 1.1:
        print(f"   • Collectivistic languages show more complex neural networks")
        print(f"     (Collective: {col_complexity:.1f} vs Individual: {ind_complexity:.1f} neurons)")
    
    return {
        'hypothesis_results': hypothesis_results,
        'support_percentage': support_percentage,
        'cultural_data': cultural_results,
        'individualistic_data': individualistic_data,
        'collectivistic_data': collectivistic_data
    }

if __name__ == "__main__":
    # Run Phase 1: Extended protocol test
    print("🧪 RUNNING COMPLETE CULTURAL-LINGUISTIC ANALYSIS")
    print("="*60)
    print()
    
    # Phase 2: Cultural pattern analysis
    cultural_results = test_phase2_cultural_patterns()
    
    # Phase 3: Hypothesis testing
    analysis = analyze_cultural_hypothesis(cultural_results)
    
    print(f"\n✅ COMPLETE ANALYSIS FINISHED")
    print(f"📊 Cultural-linguistic correlation analysis complete")
    print(f"🎯 Hypothesis support level: {analysis['support_percentage']:.1f}%")