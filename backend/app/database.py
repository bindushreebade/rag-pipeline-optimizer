import mysql.connector
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
 
from .config import settings
 
 
class ChatHistoryDB:
    """MySQL database handler for chat history storage and retrieval."""
    
    def __init__(self):
        self.config = {
            "host": settings.MYSQL_HOST,
            "user": settings.MYSQL_USER,
            "password": settings.MYSQL_PASSWORD,
            "database": settings.MYSQL_DATABASE,
        }
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = mysql.connector.connect(**self.config)
        try:
            yield conn
        finally:
            conn.close()
    
    def init_db(self):
        """Initialize database tables."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create chat_history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        question TEXT NOT NULL,
                        answer LONGTEXT NOT NULL,
                        pipeline VARCHAR(50),
                        document_name VARCHAR(255),
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        retrieved_documents LONGTEXT,
                        INDEX idx_timestamp (timestamp),
                        INDEX idx_document (document_name)
                    )
                """)
                
                # Create pipeline_comparison table for storing multi-pipeline results
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pipeline_comparison (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        question TEXT NOT NULL,
                        pipeline_a_answer LONGTEXT,
                        pipeline_b_answer LONGTEXT,
                        pipeline_c_answer LONGTEXT,
                        pipeline_d_answer LONGTEXT,
                        best_pipeline VARCHAR(10),
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        document_name VARCHAR(255),
                        INDEX idx_timestamp (timestamp),
                        INDEX idx_document (document_name)
                    )
                """)
                
                conn.commit()
                print("✓ Database tables initialized successfully")
        except mysql.connector.Error as err:
            print(f"✗ Database initialization error: {err}")
            raise
    
    def save_chat(self, question: str, answer: str, pipeline: str = None, 
                  document_name: str = None, retrieved_documents: str = None) -> int:
        """Save a single chat interaction to history."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO chat_history 
                    (question, answer, pipeline, document_name, retrieved_documents, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (question, answer, pipeline, document_name, retrieved_documents, datetime.now()))
                conn.commit()
                return cursor.lastrowid
        except mysql.connector.Error as err:
            print(f"✗ Error saving chat: {err}")
            raise
    
    def save_pipeline_comparison(self, question: str, results: Dict[str, str], 
                                 best_pipeline: str, document_name: str = None) -> int:
        """Save multi-pipeline comparison results."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO pipeline_comparison 
                    (question, pipeline_a_answer, pipeline_b_answer, pipeline_c_answer, 
                     pipeline_d_answer, best_pipeline, document_name, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    question,
                    results.get("A"),
                    results.get("B"),
                    results.get("C"),
                    results.get("D"),
                    best_pipeline,
                    document_name,
                    datetime.now()
                ))
                conn.commit()
                return cursor.lastrowid
        except mysql.connector.Error as err:
            print(f"✗ Error saving pipeline comparison: {err}")
            raise
    
    def get_chat_history(self, limit: int = 50, document_name: str = None) -> List[Dict[str, Any]]:
        """Retrieve chat history with optional filtering."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                if document_name:
                    cursor.execute("""
                        SELECT * FROM chat_history 
                        WHERE document_name = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (document_name, limit))
                else:
                    cursor.execute("""
                        SELECT * FROM chat_history 
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (limit,))
                
                return cursor.fetchall()
        except mysql.connector.Error as err:
            print(f"✗ Error retrieving chat history: {err}")
            return []
    
    def get_pipeline_comparisons(self, limit: int = 50, document_name: str = None) -> List[Dict[str, Any]]:
        """Retrieve pipeline comparison history."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                if document_name:
                    cursor.execute("""
                        SELECT * FROM pipeline_comparison 
                        WHERE document_name = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (document_name, limit))
                else:
                    cursor.execute("""
                        SELECT * FROM pipeline_comparison 
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (limit,))
                
                return cursor.fetchall()
        except mysql.connector.Error as err:
            print(f"✗ Error retrieving pipeline comparisons: {err}")
            return []
    
    def delete_chat_history(self, chat_id: int) -> bool:
        """Delete a specific chat history entry."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chat_history WHERE id = %s", (chat_id,))
                conn.commit()
                return cursor.rowcount > 0
        except mysql.connector.Error as err:
            print(f"✗ Error deleting chat history: {err}")
            return False
    
    def get_statistics(self, document_name: str = None) -> Dict[str, Any]:
        """Get statistics about chat history."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                if document_name:
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total_chats,
                            COUNT(DISTINCT DATE(timestamp)) as unique_days
                        FROM chat_history
                        WHERE document_name = %s
                    """, (document_name,))
                else:
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total_chats,
                            COUNT(DISTINCT DATE(timestamp)) as unique_days
                        FROM chat_history
                    """)
                
                result = cursor.fetchone()
                return result if result else {"total_chats": 0, "unique_days": 0}
        except mysql.connector.Error as err:
            print(f"✗ Error retrieving statistics: {err}")
            return {"total_chats": 0, "unique_days": 0}
 
 
# Global instance
db = ChatHistoryDB()
 
