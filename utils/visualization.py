import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any

class LearningVisualizer:
    def __init__(self):
        self.color_scheme = {
            'knowledge': '#1f77b4',
            'engagement': '#ff7f0e',
            'performance': '#2ca02c',
            'weak_area': '#d62728'
        }
    
    def create_student_dashboard(self, student_insights: Dict[str, Any],
                               progress_data: Dict[str, Any]) -> go.Figure:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Knowledge Progress', 'Engagement Trend',
                          'Learning Velocity', 'Performance Distribution'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "indicator"}, {"type": "box"}]]
        )
        
        if 'knowledge_growth' in progress_data:
            for concept, growth in progress_data['knowledge_growth'].items():
                if len(growth) > 1:
                    timestamps = [point['timestamp'] for point in growth]
                    knowledge_levels = [point['knowledge_level'] for point in growth]
                    
                    fig.add_trace(
                        go.Scatter(x=timestamps, y=knowledge_levels, name=concept,
                                  line=dict(width=2)),
                        row=1, col=1
                    )
        
        if 'engagement_history' in progress_data:
            engagement_data = progress_data['engagement_history']
            if len(engagement_data) > 1:
                timestamps = [point['timestamp'] for point in engagement_data]
                engagement_levels = [point['level'] for point in engagement_data]
                
                fig.add_trace(
                    go.Scatter(x=timestamps, y=engagement_levels, name='Engagement',
                              line=dict(color=self.color_scheme['engagement'], width=3)),
                    row=1, col=2
                )
        
        learning_velocity = student_insights.get('learning_velocity', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=learning_velocity,
                title={'text': "Learning Velocity"},
                gauge={'axis': {'range': [0, 10]},
                      'bar': {'color': self.color_scheme['performance']},
                      'steps': [{'range': [0, 3], 'color': "lightgray"},
                               {'range': [3, 7], 'color': "yellow"},
                               {'range': [7, 10], 'color': "green"}]}),
            row=2, col=1
        )
        
        performances = [session['performance'] for session in progress_data.get('learning_sessions', [])]
        if performances:
            fig.add_trace(
                go.Box(y=performances, name='Performance',
                      marker_color=self.color_scheme['performance']),
                row=2, col=2
            )
        
        fig.update_layout(height=600, title_text="Student Learning Dashboard")
        return fig
    
    def create_concept_network(self, concept_graph: Dict[str, Any],
                             student_knowledge: Dict[str, float]) -> go.Figure:
        nodes = []
        edges = []
        
        for concept, data in concept_graph.items():
            knowledge_level = student_knowledge.get(concept, 0)
            node_color = f'rgba(31, 119, 180, {knowledge_level})'
            
            nodes.append({
                'id': concept,
                'label': concept,
                'color': node_color,
                'size': 10 + (knowledge_level * 20)
            })
            
            for prereq in data.get('prerequisites', []):
                edges.append({
                    'from': prereq,
                    'to': concept,
                    'arrows': 'to'
                })
        
        fig = go.Figure()
        
        for edge in edges:
            fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(width=1, color='gray')))
        
        for node in nodes:
            fig.add_trace(go.Scatter(x=[], y=[], mode='markers+text', 
                                   marker=dict(size=node['size'], color=node['color']),
                                   text=node['label'], textposition="middle center"))
        
        fig.update_layout(title="Concept Knowledge Network", showlegend=False)
        return fig