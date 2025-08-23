"""  
Procesador de Archivos del Ministerio de Educación
Transforma archivos del MEN a formato CSV estándar con ID único
Procesa por separado programas universitarios y técnicos

Autor: Erika Monroy
Actualización: Procesamiento separado con filtros por municipio
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import re
import hashlib
import uuid

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ministry_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MinistryDataProcessor:
    """
    Procesador de datos del Ministerio de Educación Nacional
    Convierte archivos CSV y Excel del MEN al formato estándar de 22 campos
    """
    
    def __init__(self):
        # Nueva estructura limpia - solo datos reales
        self.target_columns = [
            'id_programa', 'nombre_programa', 'nivel_academico', 'duracion_info', 
            'formato', 'costo', 'ubicacion', 'institucion', 'codigo_programa', 
            'estado_programa', 'area_conocimiento', 'tipo_institucion'
        ]
        
        # Contador para IDs secuenciales
        self.id_counter = 1
        
        # Mapeo de tipos de certificado a nivel académico (educación técnica)
        self.cert_to_level = {
            'TÉCNICO LABORAL': 'Técnico',
            'CONOCIMIENTOS ACADÉMICOS': 'Curso',
            'AUXILIAR': 'Técnico Auxiliar'
        }
        
        # Mapeo de niveles de formación a nivel académico (educación superior)
        self.formacion_to_level = {
            'Pregrado': 'Universitario',
            'Universitaria': 'Universitario', 
            'Maestría': 'Maestría',
            'Doctorado': 'Doctorado',
            'Especialización': 'Especialización',
            'Tecnológica': 'Tecnológico'
        }
        
        # Mapeo de jornadas a formato
        self.jornada_to_format = {
            'DIURNA': 'Presencial',
            'NOCTURNA': 'Nocturno',
            'FIN DE SEMANA': 'Fines de Semana',
            'DIURNA,NOCTURNA': 'Mixta',
            'DIURNA,FIN DE SEMANA': 'Mixta',
            'NOCTURNA,FIN DE SEMANA': 'Mixta',
            'DIURNA,NOCTURNA,FIN DE SEMANA': 'Mixta'
        }
    
    def generate_sequential_id(self) -> int:
        """
        Genera un ID secuencial numérico único
        """
        current_id = self.id_counter
        self.id_counter += 1
        return current_id
    
    def load_csv_file(self, file_path: str) -> pd.DataFrame:
        """
        Carga archivo CSV del Ministerio
        """
        try:
            logger.info(f"Cargando archivo CSV: {file_path}")
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.info(f"CSV cargado exitosamente. Filas: {len(df)}, Columnas: {len(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Error cargando CSV {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def process_technical_programs(self, df: pd.DataFrame, municipality_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Procesa programas técnicos del archivo MEN_PROGRAMAS_EDUCACI_N_PARA_EL_TRABAJO_Y_EL_DESARROLLO_HUMANO
        Implementa estructura limpia según PROPUESTA_ESTRUCTURA_LIMPIA.md
        """
        logger.info("Procesando programas técnicos con estructura limpia...")
        
        # Filtrar por municipio si se especifica
        if municipality_filter:
            mask = df['Municipio'].isin(municipality_filter)
            df = df.loc[mask].copy()
            logger.info(f"Filtrado por municipios {municipality_filter}: {len(df)} registros")
        
        processed_data = []
        
        for _, row in df.iterrows():
            try:
                # Construir ubicación - solo municipio
                municipio = str(row.get('Municipio', '')).strip()
                ubicacion = municipio if municipio else 'No especificado'
                
                # Procesar jornadas con limpieza mejorada
                jornadas = str(row.get('Jornadas', '')).strip().replace('"', '').replace(' ', '')
                if not jornadas or jornadas.lower() in ['nan', 'none', '']:
                    jornadas = 'DIURNA'
                formato = self.jornada_to_format.get(jornadas, 'Presencial')
                
                # Determinar nivel académico
                tipo_cert = str(row.get('Tipo Certificado', '')).strip()
                if 'TÉCNICO LABORAL' in tipo_cert:
                    nivel_academico = 'Técnico'
                elif 'CONOCIMIENTOS ACADÉMICOS' in tipo_cert:
                    nivel_academico = 'Curso'
                else:
                    nivel_academico = 'Técnico'
                
                # Procesar área de desempeño
                area_desempeno = str(row.get('Área Desempeño', '')).strip()
                if not area_desempeno or area_desempeno.lower() in ['nan', 'none', '']:
                    area_desempeno = 'No especificada'
                
                # Procesar duración - mantener como texto con unidad
                duracion_horas = row.get('Duración Horas', 0)
                try:
                    horas = int(float(duracion_horas)) if duracion_horas and not pd.isna(duracion_horas) else 0
                    duracion_info = f"{horas} horas" if horas > 0 else "No especificada"
                except (ValueError, TypeError):
                    duracion_info = "No especificada"
                
                # Crear registro procesado según estructura limpia
                processed_row = {
                    'id_programa': self.generate_sequential_id(),
                    'nombre_programa': str(row.get('Nombre Programa', '')).strip(),
                    'nivel_academico': nivel_academico,
                    'duracion_info': duracion_info,
                    'formato': formato,
                    'costo': self.clean_cost(row.get('Costo', 0)),
                    'ubicacion': ubicacion,
                    'institucion': str(row.get('Nombre Institución', '')).strip(),
                    'codigo_programa': str(row.get('Código Programa', '')).strip(),
                    'estado_programa': str(row.get('Estado Programa', 'Activo')).strip(),
                    'area_conocimiento': area_desempeno,
                    'tipo_institucion': 'Técnica'
                }
                
                processed_data.append(processed_row)
                
            except Exception as e:
                logger.warning(f"Error procesando programa técnico: {str(e)}")
                continue
        
        result_df = pd.DataFrame(processed_data)
        logger.info(f"Programas técnicos procesados: {len(result_df)}")
        return result_df
    
    def process_university_programs(self, df: pd.DataFrame, municipality_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Procesa programas universitarios del archivo MEN_PROGRAMAS_DE_EDUCACI_N_SUPERIOR
        Implementa estructura limpia según PROPUESTA_ESTRUCTURA_LIMPIA.md
        """
        logger.info("Procesando programas universitarios con estructura limpia...")
        
        # Filtrar por municipio si se especifica
        if municipality_filter:
            mask = df['nombremunicipioinstitucion'].isin(municipality_filter)
            df = df.loc[mask].copy()
            logger.info(f"Filtrado por municipios {municipality_filter}: {len(df)} registros")
        
        processed_data = []
        
        for _, row in df.iterrows():
            try:
                # Construir ubicación - solo municipio
                municipio = str(row.get('nombremunicipioinstitucion', '')).strip()
                ubicacion = municipio if municipio else 'No especificado'
                
                # Determinar nivel académico
                nivel_formacion = str(row.get('nombrenivelformacion', '')).strip()
                nivel_academico = 'Universitario'  # Default
                for key, value in self.formacion_to_level.items():
                    if key in nivel_formacion:
                        nivel_academico = value
                        break
                
                # Procesar duración - mantener como texto descriptivo
                periodos = row.get('cantidadperiodos', 0)
                periodicidad = str(row.get('nombreperiodicidad', '')).strip()
                try:
                    if periodos and not pd.isna(periodos) and periodicidad:
                        periodos_int = int(periodos)
                        if 'Semestral' in periodicidad:
                            duracion_info = f"{periodos_int} períodos semestrales"
                        elif 'Trimestral' in periodicidad:
                            duracion_info = f"{periodos_int} períodos trimestrales"
                        elif 'Anual' in periodicidad:
                            duracion_info = f"{periodos_int} períodos anuales"
                        else:
                            duracion_info = f"{periodos_int} períodos"
                    else:
                        duracion_info = "No especificada"
                except (ValueError, TypeError):
                    duracion_info = "No especificada"
                
                # Procesar metodología y formato
                metodologia = str(row.get('nombremetodologia', 'Presencial')).strip()
                formato = metodologia
                if 'distancia' in metodologia.lower():
                    formato = 'A distancia'
                elif 'virtual' in metodologia.lower():
                    formato = 'Virtual'
                else:
                    formato = 'Presencial'
                
                # Procesar área de conocimiento
                area_conocimiento = str(row.get('nombreareaconocimiento', 'General')).strip()
                if not area_conocimiento or area_conocimiento.lower() in ['nan', 'none', '']:
                    area_conocimiento = 'General'
                
                # Determinar tipo de institución
                tipo_institucion_raw = str(row.get('nombreorigeninstitucional', 'Privada')).strip()
                if 'Oficial' in tipo_institucion_raw:
                    tipo_institucion = 'Oficial'
                else:
                    tipo_institucion = 'Privada'
                
                # Crear registro procesado según estructura limpia
                processed_row = {
                    'id_programa': self.generate_sequential_id(),
                    'nombre_programa': str(row.get('nombretituloobtenido', '')).strip(),
                    'nivel_academico': nivel_academico,
                    'duracion_info': duracion_info,
                    'formato': formato,
                    'costo': 'Consultar institución',
                    'ubicacion': ubicacion,
                    'institucion': str(row.get('nombreinstitucion', '')).strip(),
                    'codigo_programa': str(row.get('codigoprograma', '')).strip(),
                    'estado_programa': str(row.get('nombreestadoprograma', 'Activo')).strip(),
                    'area_conocimiento': area_conocimiento,
                    'tipo_institucion': tipo_institucion
                }
                
                processed_data.append(processed_row)
                
            except Exception as e:
                logger.warning(f"Error procesando programa universitario: {str(e)}")
                continue
        
        result_df = pd.DataFrame(processed_data)
        logger.info(f"Programas universitarios procesados: {len(result_df)}")
        return result_df
    
    def calculate_duration_semesters(self, duration_hours: Any) -> int:
        """
        Calcula duración en semestres basado en horas
        """
        try:
            if pd.isna(duration_hours) or duration_hours == 0:
                return 2  # Default
            
            hours = float(duration_hours)
            # Aproximación: 1 semestre = 720 horas académicas
            semesters = max(1, round(hours / 720))
            return min(semesters, 10)  # Máximo 10 semestres
        except:
            return 2
    
    def determine_difficulty_level(self, escolaridad: Optional[str], tipo_cert: Optional[str]) -> str:
        """
        Determina nivel de dificultad basado en escolaridad y tipo
        """
        if escolaridad is None or pd.isna(escolaridad) or str(escolaridad).strip() == '':
            return 'Intermedio'
        
        escolaridad_clean = str(escolaridad).strip().upper()
        tipo_clean = str(tipo_cert).strip().upper() if tipo_cert is not None and not pd.isna(tipo_cert) else ''
        
        if 'PRIMARIA' in escolaridad_clean:
            return 'Básico'
        elif 'SECUNDARIA' in escolaridad_clean:
            if 'TÉCNICO LABORAL' in tipo_clean:
                return 'Intermedio'
            else:
                return 'Básico'
        elif 'MEDIA' in escolaridad_clean:
            return 'Intermedio'
        else:
            return 'Intermedio'
    
    def clean_cost(self, cost: Any) -> int:
        """
        Limpia y convierte costo a entero
        """
        try:
            if pd.isna(cost) or cost == 0:
                return 0
            
            cost_str = str(cost).replace(',', '').replace('.', '')
            cost_clean = re.sub(r'[^0-9]', '', cost_str)
            return int(cost_clean) if cost_clean else 0
        except:
            return 0
    
    def create_general_programs(self, df_programs: pd.DataFrame) -> pd.DataFrame:
        """
        Crea lista de programas generales únicos con ID único
        Filtra solo programas técnicos, tecnológicos y universitarios
        Aplica reglas de generalización para reducir duplicados
        """
        logger.info("Creando lista de programas generales filtrados y generalizados...")
        
        # Filtrar solo niveles académicos deseados
        allowed_levels = ['Técnico', 'Tecnológico', 'Universitario']
        filtered_df = df_programs[df_programs['nivel_academico'].isin(allowed_levels)].copy()
        logger.info(f"Programas filtrados por nivel académico: {len(filtered_df)} de {len(df_programs)}")
        
        # Reglas de generalización
        generalization_rules = {
            # Ingeniería de Sistemas y variantes
            r'INGENIER[OA]\s+(DE\s+)?SISTEMAS.*': 'INGENIERO DE SISTEMAS',
            r'INGENIER[OA]\s+INFORMATICA.*': 'INGENIERO DE SISTEMAS',
            r'INGENIER[OA]\s+EN\s+SISTEMAS.*': 'INGENIERO DE SISTEMAS',
            
            # Técnicos en sistemas
            r'T[ÉE]CNICO\s+LABORAL\s+EN\s+SISTEMAS.*': 'TÉCNICO EN SISTEMAS',
            r'T[ÉE]CNICO\s+EN\s+SISTEMAS.*': 'TÉCNICO EN SISTEMAS',
            r'T[ÉE]CNICO\s+LABORAL\s+POR\s+COMPETENCIAS\s+EN\s+SISTEMAS.*': 'TÉCNICO EN SISTEMAS',
            
            # Auxiliares administrativos
            r'T[ÉE]CNICO\s+LABORAL\s+EN\s+AUXILIAR\s+ADMINISTRATIVO.*': 'TÉCNICO AUXILIAR ADMINISTRATIVO',
            r'AUXILIAR\s+ADMINISTRATIVO.*': 'TÉCNICO AUXILIAR ADMINISTRATIVO',
            r'T[ÉE]CNICO\s+LABORAL\s+POR\s+COMPETENCIAS?\s+EN\s+ASISTENCIA\s+ADMINISTRATIVA.*': 'TÉCNICO AUXILIAR ADMINISTRATIVO',
            
            # Auxiliares contables
            r'T[ÉE]CNICO\s+LABORAL\s+.*AUXILIAR\s+CONTABLE.*': 'TÉCNICO AUXILIAR CONTABLE',
            r'AUXILIAR\s+CONTABLE.*': 'TÉCNICO AUXILIAR CONTABLE',
            r'T[ÉE]CNICO\s+LABORAL\s+EN\s+CONTABILIDAD.*': 'TÉCNICO EN CONTABILIDAD',
            
            # Enfermería
            r'T[ÉE]CNICO\s+LABORAL\s+EN\s+AUXILIAR\s+EN\s+ENFERMER[ÍI]A.*': 'TÉCNICO AUXILIAR EN ENFERMERÍA',
            r'AUXILIAR\s+EN\s+ENFERMER[ÍI]A.*': 'TÉCNICO AUXILIAR EN ENFERMERÍA',
            
            # Primera infancia
            r'T[ÉE]CNICO\s+LABORAL\s+EN\s+.*PRIMERA\s+INFANCIA.*': 'TÉCNICO EN PRIMERA INFANCIA',
            r'.*PRIMERA\s+INFANCIA.*': 'TÉCNICO EN PRIMERA INFANCIA',
            
            # Peluquería y estética
            r'T[ÉE]CNICO\s+LABORAL\s+EN\s+PELUQUER[ÍI]A.*': 'TÉCNICO EN PELUQUERÍA',
            r'PELUQUER[ÍI]A.*': 'TÉCNICO EN PELUQUERÍA',
            r'T[ÉE]CNICO\s+LABORAL\s+EN\s+COSMETOLOG[ÍI]A.*': 'TÉCNICO EN COSMETOLOGÍA',
            
            # Cocina y gastronomía
            r'T[ÉE]CNICO\s+LABORAL\s+EN\s+COCINA.*': 'TÉCNICO EN COCINA',
            r'COCINERO.*': 'TÉCNICO EN COCINA',
            
            # Secretariado
            r'T[ÉE]CNICO\s+LABORAL\s+EN\s+SECRETARIADO.*': 'TÉCNICO EN SECRETARIADO',
            r'SECRETARIADO.*': 'TÉCNICO EN SECRETARIADO',
        }
        
        # Aplicar generalización
        def generalize_program_name(name):
            name_upper = str(name).upper().strip()
            
            # Filtrar programas duplicados o en proceso de cancelación
            if any(keyword in name_upper for keyword in ['(DUPLICADO)', '(EN PROCESO DE CANCELACION)', '(EN PROCESO DE CANCELACIÓN)']):
                return None  # Marcar para exclusión
            
            # Reglas específicas de consolidación
            if 'ABOGAD' in name_upper:
                return 'ABOGADO'
            
            for pattern, replacement in generalization_rules.items():
                if re.match(pattern, name_upper):
                    return replacement
            return name_upper
        
        # Aplicar generalización a los nombres usando iteración manual
        generalized_names = []
        for _, row in filtered_df.iterrows():
            generalized_name = generalize_program_name(row['nombre_programa'])
            if generalized_name is not None:  # Excluir programas marcados para exclusión
                generalized_names.append(generalized_name)
        
        # Obtener nombres únicos generalizados usando set
        unique_programs = list(set(generalized_names))
        
        general_programs = []
        for i, program_name in enumerate(sorted(unique_programs), 1):
            if program_name and str(program_name).strip():
                general_programs.append({
                    'id_programa': i,  # ID secuencial desde 1
                    'nombre_programa': str(program_name).strip()
                })
        
        result_df = pd.DataFrame(general_programs)
        logger.info(f"Programas generales creados después de filtrado y generalización: {len(result_df)}")
        return result_df
    
    def process_ministry_data(self, target_municipalities: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Procesa ambos archivos del ministerio y retorna programas filtrados y programas generales
        
        Args:
            target_municipalities: Lista de municipios para filtrar
            
        Returns:
            Tuple con (programas_filtrados, programas_generales)
        """
        logger.info("Iniciando procesamiento de datos del ministerio...")
        
        # Cargar datos técnicos
        try:
            df_technical = pd.read_csv("./MEN_PROGRAMAS_EDUCACI_N_PARA_EL_TRABAJO_Y_EL_DESARROLLO_HUMANO_20250811.csv", encoding='utf-8')
            logger.info(f"Archivo técnico cargado: {len(df_technical)} registros")
        except Exception as e:
            logger.error(f"Error cargando archivo técnico: {str(e)}")
            df_technical = pd.DataFrame()
        
        # Cargar datos universitarios
        try:
            df_university = pd.read_csv("./MEN_PROGRAMAS_DE_EDUCACI_N_SUPERIOR_20250811.csv", encoding='utf-8')
            logger.info(f"Archivo universitario cargado: {len(df_university)} registros")
        except Exception as e:
            logger.error(f"Error cargando archivo universitario: {str(e)}")
            df_university = pd.DataFrame()
        
        # Procesar programas técnicos
        technical_df = self.process_technical_programs(df_technical, target_municipalities)
        
        # Procesar programas universitarios
        university_df = self.process_university_programs(df_university, target_municipalities)
        
        # Combinar ambos datasets
        combined_df = pd.concat([technical_df, university_df], ignore_index=True)
        logger.info(f"Total de programas combinados: {len(combined_df)}")
        
        # Crear programas generales
        general_programs_df = self.create_general_programs(combined_df)
        
        return combined_df, general_programs_df
    
    def apply_filters(self, df: pd.DataFrame, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Aplica filtros específicos al DataFrame procesado
        
        Args:
            df: DataFrame a filtrar
            filters: Diccionario con filtros a aplicar
                    Ejemplo: {'ubicacion': 'Bogotá', 'nivel_academico': 'Técnico'}
        
        Returns:
            DataFrame filtrado
        """
        if filters is None or not filters:
            return df
        
        logger.info(f"Aplicando filtros: {filters}")
        filtered_df = df.copy()
        original_count = len(filtered_df)
        
        for column, filter_value in filters.items():
            if column not in filtered_df.columns:
                logger.warning(f"Columna '{column}' no encontrada. Saltando filtro.")
                continue
            
            if isinstance(filter_value, str):
                # Filtro por texto (case-insensitive, contiene)
                mask = filtered_df[column].astype(str).str.contains(
                    filter_value, case=False, na=False
                )
                filtered_df = filtered_df.loc[mask]
                logger.info(f"Filtro '{column}' contiene '{filter_value}': {len(filtered_df)} registros")
            
            elif isinstance(filter_value, list):
                # Filtro por lista de valores
                mask = filtered_df[column].isin(filter_value)
                filtered_df = filtered_df.loc[mask]
                logger.info(f"Filtro '{column}' en {filter_value}: {len(filtered_df)} registros")
            
            elif isinstance(filter_value, dict):
                # Filtros avanzados (rango, operadores)
                if 'min' in filter_value or 'max' in filter_value:
                    # Filtro por rango numérico
                    if 'min' in filter_value:
                        mask = pd.to_numeric(filtered_df[column], errors='coerce') >= filter_value['min']
                        filtered_df = filtered_df.loc[mask]
                    if 'max' in filter_value:
                        mask = pd.to_numeric(filtered_df[column], errors='coerce') <= filter_value['max']
                        filtered_df = filtered_df.loc[mask]
                    logger.info(f"Filtro '{column}' rango {filter_value}: {len(filtered_df)} registros")
        
        # Reordenar IDs después del filtrado
        if len(filtered_df) > 0:
            filtered_df = filtered_df.reset_index(drop=True)
            filtered_df['id_programa'] = range(1, len(filtered_df) + 1)
        
        logger.info(f"Filtrado completado. Antes: {original_count}, Después: {len(filtered_df)}")
        return filtered_df
    

    
    def merge_and_deduplicate(self, df_tecnico: pd.DataFrame, df_superior: pd.DataFrame) -> pd.DataFrame:
        """
        Combina y deduplica datos de ambas fuentes usando la estructura limpia
        """
        logger.info("Combinando y deduplicando datos con estructura limpia...")
        
        # Asegurar que ambos DataFrames tengan las columnas esperadas
        expected_columns = [
            'id_programa', 'nombre_programa', 'nivel_academico', 'duracion_info',
            'formato', 'costo', 'ubicacion', 'institucion', 'codigo_programa',
            'estado_programa', 'area_conocimiento', 'tipo_institucion'
        ]
        
        # Crear copias para evitar modificar los DataFrames originales
        df_tecnico_copy = df_tecnico.copy()
        df_superior_copy = df_superior.copy()
        
        # Verificar y agregar columnas faltantes en df_tecnico
        for col in expected_columns:
            if col not in df_tecnico_copy.columns:
                df_tecnico_copy[col] = 'No especificado'
        
        # Verificar y agregar columnas faltantes en df_superior
        for col in expected_columns:
            if col not in df_superior_copy.columns:
                df_superior_copy[col] = 'No especificado'
        
        # Reordenar columnas según estructura limpia
        df_tecnico_final = df_tecnico_copy[expected_columns]
        df_superior_final = df_superior_copy[expected_columns]
        
        # Combinar DataFrames
        combined_df = pd.concat([df_tecnico_final, df_superior_final], ignore_index=True)
        
        # Reordenar IDs secuencialmente después de la combinación
        combined_df = combined_df.reset_index(drop=True)
        combined_df['id_programa'] = range(1, len(combined_df) + 1)
        
        # Deduplicar por nombre de programa e institución
        before_dedup = len(combined_df)
        # Usar groupby para deduplicar de forma compatible
        combined_df = combined_df.groupby(['nombre_programa', 'institucion']).first().reset_index()
        after_dedup = len(combined_df)
        
        # Reordenar IDs nuevamente después de deduplicación
        combined_df = combined_df.reset_index(drop=True)
        combined_df['id_programa'] = range(1, len(combined_df) + 1)
        
        logger.info(f"Deduplicación completada. Antes: {before_dedup}, Después: {after_dedup}")
        
        return combined_df
    
    def export_to_csv(self, df: pd.DataFrame, output_path: str) -> bool:
        """
        Exporta DataFrame procesado a CSV
        """
        try:
            logger.info(f"Exportando datos a: {output_path}")
            
            # Asegurar que todas las columnas objetivo estén presentes
            for col in self.target_columns:
                if col not in df.columns:
                    df[col] = ''
            
            # Reordenar columnas según formato objetivo
            df_export = df[self.target_columns]
            
            # Exportar
            df_export.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Exportación exitosa. Registros exportados: {len(df_export)}")
            return True
            
        except Exception as e:
            logger.error(f"Error en exportación: {str(e)}")
            return False
    
    def process_ministry_files(self, tecnico_path: str, superior_path: str, output_path: str, 
                                 filters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Procesa ambos archivos del ministerio y genera CSV final
        
        Args:
            tecnico_path: Ruta del archivo de educación técnica
            superior_path: Ruta del archivo de educación superior
            output_path: Ruta del archivo de salida
            filters: Filtros a aplicar (opcional)
        
        Returns:
            True si el procesamiento fue exitoso, False en caso contrario
        """
        logger.info("=== INICIANDO PROCESAMIENTO DE ARCHIVOS DEL MINISTERIO ===")
        
        try:
            # Cargar archivos
            df_tecnico = self.load_csv_file(tecnico_path)
            df_superior = self.load_csv_file(superior_path)
            
            # Procesar datos
            processed_tecnico = self.process_technical_programs(df_tecnico) if not df_tecnico.empty else pd.DataFrame()
            processed_superior = self.process_university_programs(df_superior) if not df_superior.empty else pd.DataFrame()
            
            # Combinar y deduplicar
            final_df = self.merge_and_deduplicate(processed_tecnico, processed_superior)
            
            # Aplicar filtros si se especificaron
            if filters:
                final_df = self.apply_filters(final_df, filters)
            
            # Exportar resultado final
            success = self.export_to_csv(final_df, output_path)
            
            if success:
                logger.info(f"=== PROCESAMIENTO COMPLETADO EXITOSAMENTE ===")
                logger.info(f"Archivo final generado: {output_path}")
                logger.info(f"Total de programas procesados: {len(final_df)}")
                logger.info(f"Programas técnicos originales: {len(processed_tecnico)}")
                logger.info(f"Programas superiores originales: {len(processed_superior)}")
                if filters:
                    logger.info(f"Filtros aplicados: {filters}")
                return True
            else:
                logger.error("Error en la exportación final")
                return False
                
        except Exception as e:
            logger.error(f"Error crítico en procesamiento: {str(e)}")
            return False

def main():
    """
    Función principal para procesar archivos del Ministerio
    """
    try:
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Crear instancia del procesador
        processor = MinistryDataProcessor()
        
        # Procesar datos (sin filtro de municipio para obtener todos los programas)
        logger.info("Procesando todos los programas del ministerio...")
        programas_df, programas_generales_df = processor.process_ministry_data()
        
        if programas_df.empty:
            logger.error("No se pudieron procesar los datos")
            return
        
        # Deduplicar por nombre de programa e institución
        programas_df = programas_df.drop_duplicates(
            subset=['nombre_programa', 'institucion'], 
            keep='first'
        )
        
        # Guardar programas completos
        output_file = 'programas.csv'
        programas_df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Programas procesados guardados en {output_file}")
        logger.info(f"Total de programas únicos: {len(programas_df)}")
        
        # Guardar programas generales
        general_output_file = 'programas_generales.csv'
        programas_generales_df.to_csv(general_output_file, index=False, encoding='utf-8')
        logger.info(f"Programas generales guardados en {general_output_file}")
        logger.info(f"Total de programas generales únicos: {len(programas_generales_df)}")
        
        # Mostrar estadísticas
        logger.info("\n=== ESTADÍSTICAS ====")
        logger.info(f"Total de programas: {len(programas_df)}")
        logger.info(f"Programas generales únicos: {len(programas_generales_df)}")
        
        # Mostrar distribución por nivel académico
        if 'nivel_academico' in programas_df.columns:
            nivel_dist = programas_df['nivel_academico'].value_counts()
            logger.info("\nDistribución por nivel académico:")
            for nivel, count in nivel_dist.items():
                logger.info(f"  {nivel}: {count}")
        
    except Exception as e:
        logger.error(f"Error en el procesamiento principal: {str(e)}")
        raise

if __name__ == "__main__":
    main()