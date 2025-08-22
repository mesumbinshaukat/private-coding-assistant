"""
Memory management utilities for the AI Agent
Implements FAISS vector store for long-term memory and similarity search
"""

import os
import json
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import hashlib

# FAISS for vector similarity search
import faiss

# Sentence transformers for embeddings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Advanced memory management system using FAISS vector store
    
    Features:
    - Vector embeddings for semantic similarity
    - FAISS index for fast similarity search
    - Persistent storage of interactions
    - Memory consolidation and pruning
    - Context-aware retrieval
    - Episodic and semantic memory separation
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 max_memory_size: int = 10000,
                 similarity_threshold: float = 0.7):
        
        self.model_name = model_name
        self.max_memory_size = max_memory_size
        self.similarity_threshold = similarity_threshold
        
        # Model for generating embeddings
        self.embedding_model = None
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # FAISS indices for different memory types
        self.episodic_index = None  # For specific interactions
        self.semantic_index = None  # For general knowledge
        self.code_index = None     # For code-specific memories
        
        # Memory storage
        self.episodic_memories = []
        self.semantic_memories = []
        self.code_memories = []
        
        # Storage paths
        self.memory_dir = Path("memory")
        self.memory_dir.mkdir(exist_ok=True)
        
        # Memory metadata
        self.memory_metadata = {
            "total_interactions": 0,
            "last_consolidation": None,
            "memory_efficiency": 1.0
        }
    
    async def initialize(self):
        """Initialize the memory management system"""
        logger.info("Initializing MemoryManager...")
        
        try:
            # Load embedding model
            self.embedding_model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
            
            # Initialize FAISS indices
            self.episodic_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            self.semantic_index = faiss.IndexFlatIP(self.embedding_dim)
            self.code_index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Load existing memories if available
            await self._load_persistent_memory()
            
            logger.info("MemoryManager initialized successfully")
            
        except Exception as e:
            logger.error(f"MemoryManager initialization failed: {e}")
            raise
    
    async def store_interaction(self, interaction_type: str, interaction_data: Dict[str, Any]):
        """
        Store an interaction in episodic memory
        
        Args:
            interaction_type: Type of interaction (code_generation, search, reasoning)
            interaction_data: The interaction data to store
        """
        try:
            # Create memory entry
            memory_entry = {
                "id": self._generate_memory_id(interaction_data),
                "type": interaction_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": interaction_data,
                "access_count": 0,
                "importance_score": self._calculate_importance_score(interaction_type, interaction_data)
            }
            
            # Generate embedding for the interaction
            text_content = self._extract_text_for_embedding(interaction_data)
            embedding = self._generate_embedding(text_content)
            
            # Store in appropriate memory
            if interaction_type == "code_generation":
                self.code_memories.append(memory_entry)
                self.code_index.add(embedding.reshape(1, -1))
            else:
                self.episodic_memories.append(memory_entry)
                self.episodic_index.add(embedding.reshape(1, -1))
            
            # Update metadata
            self.memory_metadata["total_interactions"] += 1
            
            # Check if memory consolidation is needed
            if len(self.episodic_memories) > self.max_memory_size:
                await self._consolidate_memory()
            
            # Periodic persistence
            if self.memory_metadata["total_interactions"] % 100 == 0:
                await self._save_persistent_memory()
            
            logger.debug(f"Stored {interaction_type} interaction in memory")
            
        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")
    
    async def retrieve_similar_memories(self, 
                                      query: str, 
                                      memory_type: str = "all",
                                      top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve similar memories based on semantic similarity
        
        Args:
            query: Search query
            memory_type: Type of memory to search (episodic, semantic, code, all)
            top_k: Number of similar memories to return
            
        Returns:
            List of similar memories with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            results = []
            
            # Search episodic memory
            if memory_type in ["episodic", "all"] and len(self.episodic_memories) > 0:
                episodic_results = await self._search_memory_index(
                    query_embedding, self.episodic_index, self.episodic_memories, top_k
                )
                results.extend(episodic_results)
            
            # Search semantic memory
            if memory_type in ["semantic", "all"] and len(self.semantic_memories) > 0:
                semantic_results = await self._search_memory_index(
                    query_embedding, self.semantic_index, self.semantic_memories, top_k
                )
                results.extend(semantic_results)
            
            # Search code memory
            if memory_type in ["code", "all"] and len(self.code_memories) > 0:
                code_results = await self._search_memory_index(
                    query_embedding, self.code_index, self.code_memories, top_k
                )
                results.extend(code_results)
            
            # Sort by similarity score and return top_k
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # Update access counts
            for result in results[:top_k]:
                result["memory"]["access_count"] += 1
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return []
    
    async def store_knowledge(self, knowledge: str, category: str = "general"):
        """
        Store general knowledge in semantic memory
        
        Args:
            knowledge: Knowledge text to store
            category: Knowledge category
        """
        try:
            knowledge_entry = {
                "id": hashlib.md5(knowledge.encode()).hexdigest(),
                "category": category,
                "content": knowledge,
                "timestamp": datetime.utcnow().isoformat(),
                "access_count": 0,
                "importance_score": 0.8  # Base importance for knowledge
            }
            
            # Generate embedding
            embedding = self._generate_embedding(knowledge)
            
            # Store in semantic memory
            self.semantic_memories.append(knowledge_entry)
            self.semantic_index.add(embedding.reshape(1, -1))
            
            logger.debug(f"Stored knowledge in semantic memory: {category}")
            
        except Exception as e:
            logger.error(f"Failed to store knowledge: {e}")
    
    async def store_feedback(self, feedback: Dict[str, Any]):
        """
        Store user feedback for learning
        
        Args:
            feedback: User feedback data
        """
        try:
            feedback_entry = {
                "id": self._generate_memory_id(feedback),
                "type": "feedback",
                "timestamp": datetime.utcnow().isoformat(),
                "sentiment": self._analyze_feedback_sentiment(feedback),
                "content": feedback,
                "importance_score": 1.0  # High importance for feedback
            }
            
            # Generate embedding
            text_content = json.dumps(feedback, indent=2)
            embedding = self._generate_embedding(text_content)
            
            # Store in episodic memory (feedback is experience-based)
            self.episodic_memories.append(feedback_entry)
            self.episodic_index.add(embedding.reshape(1, -1))
            
            logger.info("User feedback stored in memory")
            
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        try:
            stats = {
                "total_memories": {
                    "episodic": len(self.episodic_memories),
                    "semantic": len(self.semantic_memories),
                    "code": len(self.code_memories)
                },
                "memory_usage": {
                    "current_size": len(self.episodic_memories) + len(self.semantic_memories) + len(self.code_memories),
                    "max_size": self.max_memory_size,
                    "utilization": (len(self.episodic_memories) + len(self.semantic_memories) + len(self.code_memories)) / self.max_memory_size
                },
                "metadata": self.memory_metadata,
                "recent_activity": await self._get_recent_memory_activity()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {}
    
    # Private methods
    def _generate_memory_id(self, data: Dict[str, Any]) -> str:
        """Generate unique ID for memory entry"""
        content = json.dumps(data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _calculate_importance_score(self, interaction_type: str, data: Dict[str, Any]) -> float:
        """Calculate importance score for memory prioritization"""
        base_scores = {
            "code_generation": 0.8,
            "search": 0.6,
            "reasoning": 0.9,
            "training": 1.0,
            "feedback": 1.0
        }
        
        base_score = base_scores.get(interaction_type, 0.5)
        
        # Adjust based on data characteristics
        if isinstance(data, dict):
            # Higher score for successful interactions
            if data.get("success", True):
                base_score += 0.1
            
            # Higher score for complex operations
            if data.get("complexity_analysis", {}).get("complexity_score", 0) > 50:
                base_score += 0.1
            
            # Higher score for high confidence results
            if data.get("confidence", 0) > 0.8:
                base_score += 0.1
        
        return min(1.0, base_score)
    
    def _extract_text_for_embedding(self, data: Dict[str, Any]) -> str:
        """Extract meaningful text from interaction data for embedding"""
        text_parts = []
        
        if isinstance(data, dict):
            # Extract key textual content
            for key, value in data.items():
                if key in ["prompt", "query", "problem", "code", "explanation", "solution"]:
                    if isinstance(value, str):
                        text_parts.append(f"{key}: {value}")
                    elif isinstance(value, dict):
                        text_parts.append(f"{key}: {json.dumps(value)}")
        
        return " ".join(text_parts) if text_parts else json.dumps(data)[:500]
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using SentenceTransformer"""
        if not text.strip():
            return np.zeros(self.embedding_dim)
        
        try:
            # Generate embedding
            embedding = self.embedding_model.encode([text])[0]
            
            # Normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.debug(f"Embedding generation failed: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    async def _search_memory_index(self, 
                                 query_embedding: np.ndarray,
                                 index: faiss.Index,
                                 memories: List[Dict],
                                 top_k: int) -> List[Dict[str, Any]]:
        """Search a FAISS index for similar memories"""
        if index.ntotal == 0:
            return []
        
        try:
            # Search FAISS index
            scores, indices = index.search(query_embedding.reshape(1, -1), min(top_k, index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(memories) and score >= self.similarity_threshold:
                    results.append({
                        "memory": memories[idx],
                        "similarity_score": float(score),
                        "relevance": "high" if score > 0.8 else "medium" if score > 0.6 else "low"
                    })
            
            return results
            
        except Exception as e:
            logger.debug(f"Index search failed: {e}")
            return []
    
    async def _consolidate_memory(self):
        """Consolidate memories by removing less important ones"""
        try:
            logger.info("Starting memory consolidation...")
            
            # Sort memories by importance and recency
            all_memories = []
            
            # Add episodic memories with source info
            for i, memory in enumerate(self.episodic_memories):
                all_memories.append({
                    "memory": memory,
                    "index": i,
                    "source": "episodic"
                })
            
            # Sort by composite score (importance + recency + access count)
            def memory_score(item):
                memory = item["memory"]
                importance = memory.get("importance_score", 0.5)
                
                # Recency score (newer is better)
                timestamp = datetime.fromisoformat(memory["timestamp"])
                age_days = (datetime.utcnow() - timestamp).days
                recency = max(0, 1 - (age_days / 365))  # Decay over a year
                
                # Access frequency
                access_frequency = min(1, memory.get("access_count", 0) / 10)
                
                return (importance * 0.5) + (recency * 0.3) + (access_frequency * 0.2)
            
            all_memories.sort(key=memory_score, reverse=True)
            
            # Keep only the top memories
            keep_count = int(self.max_memory_size * 0.8)  # Keep 80% of max size
            
            memories_to_keep = all_memories[:keep_count]
            
            # Rebuild episodic memory and index
            new_episodic_memories = []
            new_episodic_index = faiss.IndexFlatIP(self.embedding_dim)
            
            for item in memories_to_keep:
                if item["source"] == "episodic":
                    memory = item["memory"]
                    new_episodic_memories.append(memory)
                    
                    # Regenerate embedding and add to new index
                    text_content = self._extract_text_for_embedding(memory["data"])
                    embedding = self._generate_embedding(text_content)
                    new_episodic_index.add(embedding.reshape(1, -1))
            
            # Replace old memory structures
            self.episodic_memories = new_episodic_memories
            self.episodic_index = new_episodic_index
            
            # Update metadata
            self.memory_metadata["last_consolidation"] = datetime.utcnow().isoformat()
            self.memory_metadata["memory_efficiency"] = len(self.episodic_memories) / self.max_memory_size
            
            logger.info(f"Memory consolidation completed. Kept {len(self.episodic_memories)} memories")
            
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
    
    async def _save_persistent_memory(self):
        """Save memory to persistent storage"""
        try:
            # Save memory data
            memory_data = {
                "episodic_memories": self.episodic_memories,
                "semantic_memories": self.semantic_memories,
                "code_memories": self.code_memories,
                "metadata": self.memory_metadata
            }
            
            with open(self.memory_dir / "memories.json", "w") as f:
                json.dump(memory_data, f, indent=2)
            
            # Save FAISS indices
            if self.episodic_index.ntotal > 0:
                faiss.write_index(self.episodic_index, str(self.memory_dir / "episodic_index.faiss"))
            
            if self.semantic_index.ntotal > 0:
                faiss.write_index(self.semantic_index, str(self.memory_dir / "semantic_index.faiss"))
            
            if self.code_index.ntotal > 0:
                faiss.write_index(self.code_index, str(self.memory_dir / "code_index.faiss"))
            
            logger.debug("Memory saved to persistent storage")
            
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    async def _load_persistent_memory(self):
        """Load memory from persistent storage"""
        try:
            memory_file = self.memory_dir / "memories.json"
            
            if memory_file.exists():
                with open(memory_file, "r") as f:
                    memory_data = json.load(f)
                
                self.episodic_memories = memory_data.get("episodic_memories", [])
                self.semantic_memories = memory_data.get("semantic_memories", [])
                self.code_memories = memory_data.get("code_memories", [])
                self.memory_metadata = memory_data.get("metadata", self.memory_metadata)
                
                # Load FAISS indices
                episodic_index_file = self.memory_dir / "episodic_index.faiss"
                if episodic_index_file.exists():
                    self.episodic_index = faiss.read_index(str(episodic_index_file))
                
                semantic_index_file = self.memory_dir / "semantic_index.faiss"
                if semantic_index_file.exists():
                    self.semantic_index = faiss.read_index(str(semantic_index_file))
                
                code_index_file = self.memory_dir / "code_index.faiss"
                if code_index_file.exists():
                    self.code_index = faiss.read_index(str(code_index_file))
                
                logger.info(f"Loaded {len(self.episodic_memories)} episodic memories from storage")
            
        except Exception as e:
            logger.warning(f"Failed to load persistent memory: {e}")
    
    def _analyze_feedback_sentiment(self, feedback: Dict[str, Any]) -> str:
        """Simple sentiment analysis for feedback"""
        content = json.dumps(feedback).lower()
        
        positive_words = ["good", "great", "excellent", "helpful", "correct", "accurate", "useful"]
        negative_words = ["bad", "wrong", "incorrect", "unhelpful", "poor", "terrible", "useless"]
        
        positive_count = sum(1 for word in positive_words if word in content)
        negative_count = sum(1 for word in negative_words if word in content)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    async def _get_recent_memory_activity(self) -> Dict[str, Any]:
        """Get recent memory activity statistics"""
        try:
            now = datetime.utcnow()
            day_ago = now - timedelta(days=1)
            week_ago = now - timedelta(days=7)
            
            recent_activity = {
                "last_24h": 0,
                "last_7d": 0,
                "most_accessed": [],
                "recent_types": {}
            }
            
            for memory in self.episodic_memories:
                timestamp = datetime.fromisoformat(memory["timestamp"])
                
                if timestamp > day_ago:
                    recent_activity["last_24h"] += 1
                
                if timestamp > week_ago:
                    recent_activity["last_7d"] += 1
                    
                    memory_type = memory.get("type", "unknown")
                    recent_activity["recent_types"][memory_type] = recent_activity["recent_types"].get(memory_type, 0) + 1
            
            # Most accessed memories
            all_memories = self.episodic_memories + self.semantic_memories + self.code_memories
            most_accessed = sorted(all_memories, key=lambda x: x.get("access_count", 0), reverse=True)[:5]
            
            recent_activity["most_accessed"] = [
                {
                    "id": memory["id"],
                    "type": memory.get("type", "unknown"),
                    "access_count": memory.get("access_count", 0)
                }
                for memory in most_accessed
            ]
            
            return recent_activity
            
        except Exception as e:
            logger.debug(f"Failed to get recent activity: {e}")
            return {}

# Testing
if __name__ == "__main__":
    async def test_memory_manager():
        manager = MemoryManager()
        await manager.initialize()
        
        # Test storing interaction
        test_interaction = {
            "prompt": "Write a Python function to calculate factorial",
            "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
            "success": True
        }
        
        await manager.store_interaction("code_generation", test_interaction)
        
        # Test retrieval
        similar_memories = await manager.retrieve_similar_memories("factorial function", "all", 3)
        
        print(f"Found {len(similar_memories)} similar memories")
        for memory in similar_memories:
            print(f"  Similarity: {memory['similarity_score']:.3f}")
            print(f"  Type: {memory['memory']['type']}")
        
        # Test stats
        stats = await manager.get_memory_stats()
        print(f"Memory stats: {stats}")
    
    asyncio.run(test_memory_manager())
