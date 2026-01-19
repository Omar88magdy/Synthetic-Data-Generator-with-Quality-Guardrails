"""
Persona management system for synthetic review generation.
"""

import random
import yaml
from typing import Dict, List, Any, Optional
import logging

from .models import Persona, ExperienceLevel, ToneType


class PersonaManager:
    """Manages personas for review generation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.personas: List[Persona] = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self._load_personas()
    
    def _load_personas(self):
        """Load personas from configuration."""
        persona_configs = self.config.get("personas", [])
        
        for persona_config in persona_configs:
            try:
                persona = Persona(
                    role=persona_config["role"],
                    experience=ExperienceLevel(persona_config["experience"]),
                    tone=ToneType(persona_config["tone"]),
                    characteristics=persona_config["characteristics"],
                    weight=persona_config.get("weight", 1.0)
                )
                self.personas.append(persona)
                self.logger.info(f"Loaded persona: {persona.role}")
                
            except Exception as e:
                self.logger.error(f"Failed to load persona {persona_config.get('role', 'unknown')}: {e}")
    
    def get_random_persona(self) -> Persona:
        """Get a random persona based on weights."""
        if not self.personas:
            raise ValueError("No personas available")
        
        weights = [persona.weight for persona in self.personas]
        selected_persona = random.choices(self.personas, weights=weights, k=1)[0]
        
        self.logger.debug(f"Selected persona: {selected_persona.role}")
        return selected_persona
    
    def get_persona_by_role(self, role: str) -> Optional[Persona]:
        """Get a persona by role name."""
        for persona in self.personas:
            if persona.role.lower() == role.lower():
                return persona
        return None
    
    def get_all_personas(self) -> List[Persona]:
        """Get all available personas."""
        return self.personas.copy()
    
    def get_personas_by_experience(self, experience: ExperienceLevel) -> List[Persona]:
        """Get personas by experience level."""
        return [p for p in self.personas if p.experience == experience]
    
    def get_personas_by_tone(self, tone: ToneType) -> List[Persona]:
        """Get personas by tone."""
        return [p for p in self.personas if p.tone == tone]
    
    def validate_persona_distribution(self, generated_reviews: List[Dict]) -> Dict[str, Any]:
        """Validate that generated reviews follow the expected persona distribution."""
        if not generated_reviews:
            return {"status": "error", "message": "No reviews to validate"}
        
        # Count persona usage
        persona_counts = {}
        total_reviews = len(generated_reviews)
        
        for review in generated_reviews:
            if "persona" in review and review["persona"]:
                role = review["persona"]["role"]
                persona_counts[role] = persona_counts.get(role, 0) + 1
        
        # Calculate actual vs expected distribution
        validation_results = {
            "status": "success",
            "total_reviews": total_reviews,
            "persona_distribution": {},
            "deviations": {},
            "max_deviation": 0.0
        }
        
        for persona in self.personas:
            expected_count = int(total_reviews * persona.weight)
            actual_count = persona_counts.get(persona.role, 0)
            expected_ratio = persona.weight
            actual_ratio = actual_count / total_reviews if total_reviews > 0 else 0
            
            deviation = abs(expected_ratio - actual_ratio)
            
            validation_results["persona_distribution"][persona.role] = {
                "expected_count": expected_count,
                "actual_count": actual_count,
                "expected_ratio": expected_ratio,
                "actual_ratio": actual_ratio,
                "deviation": deviation
            }
            
            validation_results["deviations"][persona.role] = deviation
            validation_results["max_deviation"] = max(
                validation_results["max_deviation"], 
                deviation
            )
        
        # Check if distribution is acceptable (within 10% deviation)
        if validation_results["max_deviation"] > 0.1:
            validation_results["status"] = "warning"
            validation_results["message"] = f"High persona distribution deviation: {validation_results['max_deviation']:.2%}"
        
        return validation_results
    
    def create_custom_persona(self, role: str, experience: str, tone: str, 
                            characteristics: List[str], weight: float = 1.0) -> Persona:
        """Create a custom persona."""
        try:
            persona = Persona(
                role=role,
                experience=ExperienceLevel(experience),
                tone=ToneType(tone),
                characteristics=characteristics,
                weight=weight
            )
            return persona
        except ValueError as e:
            raise ValueError(f"Invalid persona parameters: {e}")
    
    def add_persona(self, persona: Persona):
        """Add a persona to the manager."""
        self.personas.append(persona)
        self.logger.info(f"Added persona: {persona.role}")
    
    def remove_persona(self, role: str) -> bool:
        """Remove a persona by role name."""
        for i, persona in enumerate(self.personas):
            if persona.role.lower() == role.lower():
                removed = self.personas.pop(i)
                self.logger.info(f"Removed persona: {removed.role}")
                return True
        return False
    
    def get_persona_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded personas."""
        if not self.personas:
            return {"count": 0}
        
        experience_counts = {}
        tone_counts = {}
        total_weight = 0
        
        for persona in self.personas:
            exp = persona.experience.value
            tone = persona.tone.value
            
            experience_counts[exp] = experience_counts.get(exp, 0) + 1
            tone_counts[tone] = tone_counts.get(tone, 0) + 1
            total_weight += persona.weight
        
        return {
            "count": len(self.personas),
            "total_weight": total_weight,
            "experience_distribution": experience_counts,
            "tone_distribution": tone_counts,
            "roles": [p.role for p in self.personas]
        }


def load_personas_from_yaml(yaml_file: str) -> PersonaManager:
    """Load personas from a YAML configuration file."""
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return PersonaManager(config)
