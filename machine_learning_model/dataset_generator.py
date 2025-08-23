#!/usr/bin/env python3
"""
Generador de Dataset Dimensional Simplificado para TalentAI

Este script genera un dataset sintético simple y efectivo:
- Rangos de dimensiones 1.0-5.0
- área_conocimiento como ID numérico (1-30)
- Perfiles diferenciados de estudiantes
- Sin correlaciones complejas para mayor simplicidad

Autor: TalentAI Team - Versión Simplificada
Fecha: 2024
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DimensionalDatasetGenerator:
    """
    Generador de dataset sintético mejorado basado en arquitectura de 8 dimensiones.
    Implementa correlaciones realistas, perfiles diferenciados y parametrización configurable.
    """
    
    def __init__(self, data_path: str = "data",
                 icfes_base_range: tuple = (240, 280),
                 icfes_strong_range: tuple = (340, 400), 
                 icfes_weak_range: tuple = (200, 240),
                 icfes_std: float = 35,
                 dim_base_range: tuple = (2.5, 3.1),
                 dim_strong_range: tuple = (3.8, 4.5),
                 dim_weak_range: tuple = (1.5, 2.2),
                 dim_std: float = 0.4,
                 hybrid_probability: float = 0.08):
        """
        Inicializa el generador con parámetros configurables.
        
        Args:
            data_path: Ruta a los archivos CSV base
            icfes_base_range: Rango (min, max) para medias de puntajes ICFES neutros
            icfes_strong_range: Rango (min, max) para medias de puntajes ICFES fuertes
            icfes_weak_range: Rango (min, max) para medias de puntajes ICFES débiles
            icfes_std: Desviación estándar para puntajes ICFES
            dim_base_range: Rango (min, max) para medias de dimensiones neutras
            dim_strong_range: Rango (min, max) para medias de dimensiones fuertes
            dim_weak_range: Rango (min, max) para medias de dimensiones débiles
            dim_std: Desviación estándar para dimensiones
            hybrid_probability: Probabilidad de generar perfiles híbridos
        """
        self.data_path = Path(data_path)
        self.areas_data = None
        
        # Parámetros configurables para mayor flexibilidad
        self.icfes_base_range = icfes_base_range
        self.icfes_strong_range = icfes_strong_range
        self.icfes_weak_range = icfes_weak_range
        self.icfes_std = icfes_std
        self.dim_base_range = dim_base_range
        self.dim_strong_range = dim_strong_range
        self.dim_weak_range = dim_weak_range
        self.dim_std = dim_std
        self.hybrid_probability = hybrid_probability
        
        # Correlaciones REALISTAS entre dimensiones y áreas de conocimiento
        # Nuevo formato: strong_dims, weak_dims, strong_icfes, weak_icfes
        self.area_dimension_strengths = {
            # STEM Areas (1-12)
            1: {  # Administración
                'strong_dims': [1, 6, 7, 8], 'weak_dims': [3, 5],
                'strong_icfes': ['matematicas', 'sociales_ciudadanas'], 'weak_icfes': ['ciencias_naturales']
            },
            2: {  # Finanzas
                'strong_dims': [1, 6, 7], 'weak_dims': [3, 4, 5],
                'strong_icfes': ['matematicas'], 'weak_icfes': ['ciencias_naturales', 'sociales_ciudadanas']
            },
            8: {  # Sistemas
                'strong_dims': [1, 3, 7, 8], 'weak_dims': [4, 5],
                'strong_icfes': ['matematicas', 'ciencias_naturales'], 'weak_icfes': ['sociales_ciudadanas']
            },
            9: {  # Redes
                'strong_dims': [1, 3, 7], 'weak_dims': [4, 5],
                'strong_icfes': ['matematicas', 'ciencias_naturales'], 'weak_icfes': ['sociales_ciudadanas']
            },
            10: {  # Ing. Civil
                'strong_dims': [1, 3, 7], 'weak_dims': [4, 5],
                'strong_icfes': ['matematicas', 'ciencias_naturales'], 'weak_icfes': ['sociales_ciudadanas']
            },
            11: {  # Ing. Industrial
                'strong_dims': [1, 6, 7, 8], 'weak_dims': [4, 5],
                'strong_icfes': ['matematicas'], 'weak_icfes': ['ciencias_naturales']
            },
            12: {  # Electrónica
                'strong_dims': [1, 3, 7], 'weak_dims': [4, 5],
                'strong_icfes': ['matematicas', 'ciencias_naturales'], 'weak_icfes': ['sociales_ciudadanas']
            },
            
            # Salud (13-15)
            13: {  # Enfermería
                'strong_dims': [2, 3, 6, 8], 'weak_dims': [1, 5],
                'strong_icfes': ['lectura_critica', 'ciencias_naturales'], 'weak_icfes': ['matematicas']
            },
            14: {  # Salud Pública
                'strong_dims': [2, 3, 4, 8], 'weak_dims': [1, 5],
                'strong_icfes': ['lectura_critica', 'ciencias_naturales', 'sociales_ciudadanas'], 'weak_icfes': ['matematicas']
            },
            15: {  # Farmacia
                'strong_dims': [1, 3, 7], 'weak_dims': [4, 5],
                'strong_icfes': ['matematicas', 'ciencias_naturales'], 'weak_icfes': ['sociales_ciudadanas']
            },
            
            # Educación (16-17)
            16: {  # Educación
                'strong_dims': [2, 4, 6, 8], 'weak_dims': [1, 3],
                'strong_icfes': ['lectura_critica', 'sociales_ciudadanas'], 'weak_icfes': ['matematicas', 'ciencias_naturales']
            },
            17: {  # Primera Infancia
                'strong_dims': [2, 4, 5, 8], 'weak_dims': [1, 3],
                'strong_icfes': ['lectura_critica', 'sociales_ciudadanas'], 'weak_icfes': ['matematicas', 'ciencias_naturales']
            },
            
            # Humanidades y Sociales (18-19)
            18: {  # Psicología
                'strong_dims': [2, 4, 7, 8], 'weak_dims': [1, 3],
                'strong_icfes': ['lectura_critica', 'sociales_ciudadanas'], 'weak_icfes': ['matematicas', 'ciencias_naturales']
            },
            19: {  # Seguridad
                'strong_dims': [2, 4, 6], 'weak_dims': [3, 5],
                'strong_icfes': ['lectura_critica', 'sociales_ciudadanas'], 'weak_icfes': ['ciencias_naturales']
            },
            
            # Arte y Diseño (20-22)
            20: {  # Arte
                'strong_dims': [2, 5, 7], 'weak_dims': [1, 3],
                'strong_icfes': ['lectura_critica'], 'weak_icfes': ['matematicas', 'ciencias_naturales']
            },
            21: {  # Música
                'strong_dims': [2, 5], 'weak_dims': [1, 3],
                'strong_icfes': ['lectura_critica'], 'weak_icfes': ['matematicas', 'ciencias_naturales']
            },
            22: {  # Diseño Gráfico
                'strong_dims': [2, 5, 7], 'weak_dims': [1, 3],
                'strong_icfes': ['lectura_critica'], 'weak_icfes': ['matematicas', 'ciencias_naturales']
            },
            
            # Agropecuario y Ambiental (23-24)
            23: {  # Agricultura
                'strong_dims': [1, 3, 8], 'weak_dims': [4, 5],
                'strong_icfes': ['matematicas', 'ciencias_naturales'], 'weak_icfes': ['sociales_ciudadanas']
            },
            24: {  # Medio Ambiente
                'strong_dims': [3, 4, 7, 8], 'weak_dims': [1, 5],
                'strong_icfes': ['ciencias_naturales', 'sociales_ciudadanas'], 'weak_icfes': ['matematicas']
            },
            
            # Logística y Técnico (25-27)
            25: {  # Logística
                'strong_dims': [1, 6, 8], 'weak_dims': [3, 5],
                'strong_icfes': ['matematicas'], 'weak_icfes': ['ciencias_naturales']
            },
            26: {  # Mecánica
                'strong_dims': [1, 3, 8], 'weak_dims': [4, 5],
                'strong_icfes': ['matematicas', 'ciencias_naturales'], 'weak_icfes': ['sociales_ciudadanas']
            },
            27: {  # Oficios Técnicos
                'strong_dims': [1, 3, 8], 'weak_dims': [4, 5],
                'strong_icfes': ['matematicas'], 'weak_icfes': ['sociales_ciudadanas']
            },
            
            # Otros (28-30)
            28: {  # Idiomas
                'strong_dims': [2, 4], 'weak_dims': [1, 3, 5],
                'strong_icfes': ['lectura_critica', 'ingles'], 'weak_icfes': ['matematicas', 'ciencias_naturales']
            },
            29: {  # Emprendimiento
                'strong_dims': [2, 6, 7], 'weak_dims': [3, 4],
                'strong_icfes': ['lectura_critica'], 'weak_icfes': ['ciencias_naturales']
            },
            30: {  # Calidad
                'strong_dims': [1, 6, 7], 'weak_dims': [4, 5],
                'strong_icfes': ['matematicas'], 'weak_icfes': ['sociales_ciudadanas']
            }
        }
        
        # Perfiles diferenciados de estudiantes
        self.student_profiles = {
            'stem_fuerte': {
                'weight': 0.25,
                'icfes_boost': {'matematicas': 0.8, 'ciencias_naturales': 0.7, 'lectura_critica': 0.3, 'sociales_ciudadanas': 0.2, 'ingles': 0.4},
                'preferred_areas': [8, 9, 10, 11, 12, 23, 24, 26, 27, 30]
            },
            'humanistico': {
                'weight': 0.20,
                'icfes_boost': {'lectura_critica': 0.8, 'sociales_ciudadanas': 0.7, 'ingles': 0.6, 'matematicas': 0.2, 'ciencias_naturales': 0.3},
                'preferred_areas': [16, 17, 18, 20, 21, 22, 28]
            },
            'equilibrado': {
                'weight': 0.30,
                'icfes_boost': {'matematicas': 0.5, 'lectura_critica': 0.5, 'ciencias_naturales': 0.5, 'sociales_ciudadanas': 0.5, 'ingles': 0.5},
                'preferred_areas': [1, 2, 13, 14, 15, 19, 25, 29]
            },
            'creativo_artistico': {
                'weight': 0.15,
                'icfes_boost': {'lectura_critica': 0.6, 'ingles': 0.5, 'sociales_ciudadanas': 0.4, 'matematicas': 0.3, 'ciencias_naturales': 0.2},
                'preferred_areas': [20, 21, 22, 3, 4, 5, 6, 7]
            },
            'bajo_rendimiento': {
                'weight': 0.10,
                'icfes_boost': {'matematicas': 0.2, 'lectura_critica': 0.3, 'ciencias_naturales': 0.2, 'sociales_ciudadanas': 0.3, 'ingles': 0.2},
                'preferred_areas': list(range(1, 31))  # Cualquier área
            }
        }
        
    def load_data(self) -> None:
        """
        Carga el archivo de áreas de conocimiento.
        """
        try:
            logger.info("Cargando datos de áreas de conocimiento...")
            self.areas_data = pd.read_csv(self.data_path / "general/areas_conocimiento.csv")
            logger.info(f"Áreas cargadas: {len(self.areas_data)} registros")
            
        except Exception as e:
            logger.error(f"Error cargando datos: {e}")
            raise
    
    def generate_enhanced_icfes_scores(self, area_id: int) -> Dict[str, float]:
        """
        Genera puntajes ICFES mejorados basados en el área de conocimiento con variabilidad.
        
        Args:
            area_id: ID del área de conocimiento (1-30)
            
        Returns:
            Diccionario con puntajes ICFES (150-500 escala)
        """
        area_profile = self.area_dimension_strengths.get(area_id, {
            'strong_dims': [], 'weak_dims': [], 'strong_icfes': [], 'weak_icfes': []
        })
        
        # Generar medias variables para este estudiante
        strong_mean = np.random.uniform(*self.icfes_strong_range)
        base_mean = np.random.uniform(*self.icfes_base_range)
        weak_mean = np.random.uniform(*self.icfes_weak_range)
        
        subjects = ['matematicas', 'lectura_critica', 'ciencias_naturales', 'sociales_ciudadanas', 'ingles']
        icfes_scores = {}
        
        for subject in subjects:
            # Determinar tipo de materia para esta área
            if subject in area_profile['strong_icfes']:
                mean_score = strong_mean
            elif subject in area_profile['weak_icfes']:
                mean_score = weak_mean
            else:
                mean_score = base_mean
            
            # Generar puntaje con variabilidad
            score = np.random.normal(mean_score, self.icfes_std)
            
            # Limitar al rango ICFES válido
            icfes_scores[subject] = round(max(150, min(500, score)), 1)
            
        return icfes_scores
    
    def calculate_enhanced_dimension_scores(self, icfes_scores: Dict[str, float], area_id: int, profile_type: str) -> List[float]:
        """
        Calcula los puntajes de las 8 dimensiones con variabilidad mejorada y anti-correlaciones.
        
        Args:
            icfes_scores: Diccionario con puntajes ICFES
            area_id: ID del área de conocimiento (1-30)
            profile_type: Tipo de perfil del estudiante
            
        Returns:
            Lista con 8 puntajes dimensionales (1.0-5.0)
        """
        dimension_scores = []
        area_profile = self.area_dimension_strengths.get(area_id, {
            'strong_dims': [], 'weak_dims': [], 'strong_icfes': [], 'weak_icfes': []
        })
        
        # Generar medias variables para este estudiante
        strong_mean = np.random.uniform(*self.dim_strong_range)
        base_mean = np.random.uniform(*self.dim_base_range)
        weak_mean = np.random.uniform(*self.dim_weak_range)
        
        # Promedio ICFES normalizado para influencia ligera
        icfes_avg = sum(icfes_scores.values()) / len(icfes_scores)
        icfes_normalized = (icfes_avg - 250) / 150  # Normalizar aproximadamente
        
        # Calcular factor de anti-correlación basado en fortalezas
        strong_dims = area_profile['strong_dims']
        weak_dims = area_profile['weak_dims']
        
        # Si hay dimensiones fuertes, calcular su promedio para anti-correlación
        strong_factor = 0
        if strong_dims:
            # Pre-calcular puntajes de dimensiones fuertes para anti-correlación
            strong_scores = []
            for dim_idx in strong_dims:
                score = np.random.normal(strong_mean + icfes_normalized * 0.2, self.dim_std)
                strong_scores.append(max(1.0, min(5.0, score)))
            strong_factor = np.mean(strong_scores) - 3.0  # Factor centrado en 0
        
        for dim_idx in range(1, 9):  # Dimensiones 1-8
            # Determinar tipo de dimensión para esta área
            if dim_idx in strong_dims:
                mean_score = strong_mean
                anti_corr_factor = 0  # No auto-correlación
            elif dim_idx in weak_dims:
                mean_score = weak_mean
                # Anti-correlación: si las fuertes son altas, las débiles tienden a ser más bajas
                anti_corr_factor = -strong_factor * 0.3  # Factor de anti-correlación moderado
            else:
                mean_score = base_mean
                anti_corr_factor = -strong_factor * 0.1  # Ligera influencia en dimensiones base
            
            # Influencia ligera del rendimiento ICFES
            icfes_influence = icfes_normalized * 0.2
            
            # Generar puntaje con variabilidad y anti-correlación
            score = np.random.normal(mean_score + icfes_influence + anti_corr_factor, self.dim_std)
            
            # Asegurar rango 1.0-5.0
            final_score = max(1.0, min(5.0, score))
            dimension_scores.append(round(final_score, 1))
        
        return dimension_scores
    

    
    def select_student_profile(self) -> str:
        """
        Selecciona un perfil de estudiante basado en las probabilidades.
        
        Returns:
            Tipo de perfil seleccionado
        """
        profiles = list(self.student_profiles.keys())
        weights = [self.student_profiles[p]['weight'] for p in profiles]
        return np.random.choice(profiles, p=weights)
    
    def select_area_for_profile(self, profile_type: str) -> int:
        """
        Selecciona un área de conocimiento apropiada para el perfil.
        
        Args:
            profile_type: Tipo de perfil del estudiante
            
        Returns:
            ID del área de conocimiento (1-30)
        """
        preferred_areas = self.student_profiles[profile_type]['preferred_areas']
        
        # 70% probabilidad de área preferida, 30% cualquier área
        if np.random.random() < 0.7:
            return np.random.choice(preferred_areas)
        else:
            return np.random.randint(1, 31)
    
    def generate_hybrid_profile(self) -> Dict:
        """
        Genera un perfil híbrido combinando fortalezas de múltiples áreas.
        
        Returns:
            Diccionario con perfil híbrido combinado
        """
        # Seleccionar 2-3 áreas aleatorias para combinar
        num_areas = np.random.choice([2, 3], p=[0.7, 0.3])
        selected_areas = np.random.choice(list(self.area_dimension_strengths.keys()), 
                                        size=num_areas, replace=False)
        
        # Combinar fortalezas y debilidades
        combined_strong_dims = set()
        combined_weak_dims = set()
        combined_strong_icfes = set()
        combined_weak_icfes = set()
        
        for area_id in selected_areas:
            area_profile = self.area_dimension_strengths[area_id]
            # Tomar algunas fortalezas de cada área (no todas para evitar super-estudiantes)
            strong_sample = np.random.choice(area_profile['strong_dims'], 
                                           size=min(2, len(area_profile['strong_dims'])), 
                                           replace=False) if area_profile['strong_dims'] else []
            combined_strong_dims.update(strong_sample)
            
            # Las debilidades se mantienen más conservadoras
            if area_profile['weak_dims']:
                weak_sample = np.random.choice(area_profile['weak_dims'], 
                                             size=1, replace=False)
                combined_weak_dims.update(weak_sample)
            
            # Combinar fortalezas ICFES
            if area_profile['strong_icfes']:
                icfes_sample = np.random.choice(area_profile['strong_icfes'], 
                                              size=min(2, len(area_profile['strong_icfes'])), 
                                              replace=False)
                combined_strong_icfes.update(icfes_sample)
        
        return {
            'strong_dims': list(combined_strong_dims),
            'weak_dims': list(combined_weak_dims),
            'strong_icfes': list(combined_strong_icfes),
            'weak_icfes': list(combined_weak_icfes),
            'is_hybrid': True,
            'source_areas': list(selected_areas)
        }
    
    def generate_hybrid_icfes_scores(self, hybrid_profile: Dict) -> Dict[str, float]:
        """
        Genera puntajes ICFES para perfiles híbridos.
        
        Args:
            hybrid_profile: Perfil híbrido generado
            
        Returns:
            Diccionario con puntajes ICFES
        """
        # Generar medias variables para este estudiante
        strong_mean = np.random.uniform(*self.icfes_strong_range)
        base_mean = np.random.uniform(*self.icfes_base_range)
        weak_mean = np.random.uniform(*self.icfes_weak_range)
        
        subjects = ['matematicas', 'lectura_critica', 'ciencias_naturales', 'sociales_ciudadanas', 'ingles']
        icfes_scores = {}
        
        for subject in subjects:
            # Determinar tipo de materia para este perfil híbrido
            if subject in hybrid_profile['strong_icfes']:
                mean_score = strong_mean
            elif subject in hybrid_profile['weak_icfes']:
                mean_score = weak_mean
            else:
                mean_score = base_mean
            
            # Generar puntaje con variabilidad
            score = np.random.normal(mean_score, self.icfes_std)
            
            # Limitar al rango ICFES válido
            icfes_scores[subject] = round(max(150, min(500, score)), 1)
            
        return icfes_scores
    
    def calculate_hybrid_dimension_scores(self, hybrid_profile: Dict, icfes_scores: Dict[str, float]) -> List[float]:
        """
        Calcula los puntajes de las 8 dimensiones para perfiles híbridos.
        
        Args:
            hybrid_profile: Perfil híbrido generado
            icfes_scores: Diccionario con puntajes ICFES
            
        Returns:
            Lista con 8 puntajes dimensionales (1.0-5.0)
        """
        dimension_scores = []
        
        # Generar medias variables para este estudiante
        strong_mean = np.random.uniform(*self.dim_strong_range)
        base_mean = np.random.uniform(*self.dim_base_range)
        weak_mean = np.random.uniform(*self.dim_weak_range)
        
        # Promedio ICFES normalizado para influencia ligera
        icfes_avg = sum(icfes_scores.values()) / len(icfes_scores)
        icfes_normalized = (icfes_avg - 250) / 150  # Normalizar aproximadamente
        
        # Calcular factor de anti-correlación basado en fortalezas híbridas
        strong_dims = hybrid_profile['strong_dims']
        weak_dims = hybrid_profile['weak_dims']
        
        # Factor de anti-correlación más suave para híbridos
        strong_factor = 0
        if strong_dims:
            strong_scores = []
            for dim_idx in strong_dims:
                score = np.random.normal(strong_mean + icfes_normalized * 0.2, self.dim_std)
                strong_scores.append(max(1.0, min(5.0, score)))
            strong_factor = np.mean(strong_scores) - 3.0
        
        for dim_idx in range(1, 9):  # Dimensiones 1-8
            # Determinar tipo de dimensión para este perfil híbrido
            if dim_idx in strong_dims:
                mean_score = strong_mean
                anti_corr_factor = 0
            elif dim_idx in weak_dims:
                mean_score = weak_mean
                # Anti-correlación más suave para híbridos
                anti_corr_factor = -strong_factor * 0.2
            else:
                mean_score = base_mean
                anti_corr_factor = -strong_factor * 0.05
            
            # Influencia ligera del rendimiento ICFES
            icfes_influence = icfes_normalized * 0.2
            
            # Generar puntaje con variabilidad y anti-correlación
            score = np.random.normal(mean_score + icfes_influence + anti_corr_factor, self.dim_std)
            
            # Asegurar rango 1.0-5.0
            final_score = max(1.0, min(5.0, score))
            dimension_scores.append(round(final_score, 1))
        
        return dimension_scores
    
    def generate_synthetic_student(self) -> Dict:
        """
        Genera un estudiante sintético con perfil diferenciado o híbrido.

        Returns:
            Diccionario con datos del estudiante
        """
        # Decidir si generar perfil híbrido
        is_hybrid = np.random.random() < self.hybrid_probability
        
        if is_hybrid:
            # Generar perfil híbrido
            hybrid_profile = self.generate_hybrid_profile()
            area_id = hybrid_profile['source_areas'][0]  # Usar primera área como referencia
            profile_type = 'hibrido'
            
            # Generar puntajes ICFES usando perfil híbrido
            icfes_scores = self.generate_hybrid_icfes_scores(hybrid_profile)
            
            # Generar puntajes de dimensiones usando perfil híbrido
            dimension_scores = self.calculate_hybrid_dimension_scores(hybrid_profile, icfes_scores)
        else:
            # Generar perfil normal
            profile_type = self.select_student_profile()
            area_id = self.select_area_for_profile(profile_type)
            
            # Generar puntajes ICFES simples
            icfes_scores = self.generate_enhanced_icfes_scores(area_id)
            
            # Calcular puntajes dimensionales
            dimension_scores = self.calculate_enhanced_dimension_scores(icfes_scores, area_id, profile_type)
        
        # Crear registro del estudiante
        student_data = {
            'matematicas': icfes_scores['matematicas'],
            'lectura_critica': icfes_scores['lectura_critica'],
            'ciencias_naturales': icfes_scores['ciencias_naturales'],
            'sociales_ciudadanas': icfes_scores['sociales_ciudadanas'],
            'ingles': icfes_scores['ingles'],
            'dimension_1_logico_matematico': dimension_scores[0],
            'dimension_2_comprension_comunicacion': dimension_scores[1],
            'dimension_3_pensamiento_cientifico': dimension_scores[2],
            'dimension_4_analisis_social_humanistico': dimension_scores[3],
            'dimension_5_creatividad_innovacion': dimension_scores[4],
            'dimension_6_liderazgo_trabajo_equipo': dimension_scores[5],
            'dimension_7_pensamiento_critico': dimension_scores[6],
            'dimension_8_adaptabilidad_aprendizaje': dimension_scores[7],
            'area_conocimiento': area_id,  # ID NUMÉRICO 1-30
            'perfil_tipo': profile_type  # Incluir tipo de perfil
        }
        
        return student_data
    
    def generate_dataset(self, num_students: int = 10000) -> pd.DataFrame:
        """
        Genera el dataset completo con estudiantes sintéticos.
        
        Args:
            num_students: Número de estudiantes a generar
            
        Returns:
            DataFrame con el dataset generado
        """
        logger.info(f"Generando dataset con {num_students} estudiantes...")
        
        students_data = []
        
        for i in range(num_students):
            if (i + 1) % 1000 == 0:
                logger.info(f"Progreso: {i + 1}/{num_students} estudiantes generados")
            
            student = self.generate_synthetic_student()
            students_data.append(student)
        
        df = pd.DataFrame(students_data)
        logger.info("Dataset generado exitosamente")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = "dataset_estudiantes.csv", for_training: bool = False) -> None:
        """
        Guarda el dataset en un archivo CSV.
        
        Args:
            df: DataFrame con el dataset
            filename: Nombre del archivo de salida
            for_training: Si True, excluye la columna 'perfil_tipo' para entrenamiento
        """
        if for_training and 'perfil_tipo' in df.columns:
            df_save = df.drop('perfil_tipo', axis=1)
            logger.info("Dataset preparado para entrenamiento (sin columna 'perfil_tipo')")
        else:
            df_save = df.copy()
            
        output_path = Path(filename)
        df_save.to_csv(output_path, index=False)
        logger.info(f"Dataset guardado en: {output_path.absolute()}")
    
    def validate_dataset(self, df: pd.DataFrame) -> None:
        """
        Valida el dataset generado.
        
        Args:
            df: DataFrame con el dataset generado
        """
        logger.info("\n=== VALIDACIÓN DEL DATASET ===")
        
        # Validar rangos
        logger.info(f"\n=== VALIDACIÓN DE RANGOS ===")
        dim_cols = [col for col in df.columns if col.startswith('dimension_')]
        for col in dim_cols:
            min_val, max_val = df[col].min(), df[col].max()
            logger.info(f"{col}: {min_val:.1f} - {max_val:.1f}")
        
        # Distribución por área
        logger.info(f"\n=== DISTRIBUCIÓN POR ÁREA ===")
        area_dist = df['area_conocimiento'].value_counts().sort_index()
        logger.info(f"Áreas representadas: {len(area_dist)}/30")
        logger.info(f"Estudiantes por área - Min: {area_dist.min()}, Max: {area_dist.max()}, Promedio: {np.mean(area_dist):.1f}")
        
        # Estadísticas básicas
        logger.info(f"\n=== ESTADÍSTICAS BÁSICAS ===")
        logger.info(f"Total estudiantes: {len(df)}")
        icfes_cols = ['matematicas', 'lectura_critica', 'ciencias_naturales', 'sociales_ciudadanas', 'ingles']
        promedio_icfes = df[icfes_cols].stack().mean()
        promedio_dimensiones = df[dim_cols].stack().mean()
        logger.info(f"Promedio ICFES: {promedio_icfes:.1f}")
        logger.info(f"Promedio Dimensiones: {promedio_dimensiones:.1f}")

def main():
    """
    Función principal para generar el dataset dimensional corregido.
    """
    logger.info("=== GENERADOR DE DATASET DIMENSIONAL CORREGIDO ===")
    logger.info("Iniciando generación con correlaciones realistas...")
    
    # Crear generador
    generator = DimensionalDatasetGenerator()
    
    # Cargar datos
    generator.load_data()
    
    # Generar dataset
    df = generator.generate_dataset(num_students=20000)
    
    # Validar dataset
    generator.validate_dataset(df)
    
    # Guardar dataset sin perfil_tipo para entrenamiento
    generator.save_dataset(df, for_training=True)
    
    logger.info("\n🎯 Misión completada exitosamente!")
    logger.info("Dataset dimensional simplificado generado con:")
    logger.info("✅ Rangos dimensionales correctos (1.0-5.0)")
    logger.info("✅ Áreas como IDs numéricos (1-30)")
    logger.info("✅ Perfiles diferenciados de estudiantes")
    logger.info("✅ Generación simple y eficiente")
    logger.info("Strong with simple and effective data, this implementation has become! 🌟")

if __name__ == "__main__":
    main()