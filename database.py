import os
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

DEFAULT_DB_PATH = "healthcare_audit.db"

def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Establishes and returns a connection to the SQLite database.
    Enables foreign keys support.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    # Return rows as dictionary-like objects for easier handling
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initializes the SQLite database tables if they do not exist.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # Create patients table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            dob TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name, dob)
        );
    """)
    
    # Create tickets table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            status TEXT NOT NULL, -- e.g., 'Open', 'Passed', 'Failed'
            referral_test TEXT,
            invoice_total REAL,
            audit_results TEXT, -- JSON string of the 6-point checklist results
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE
        );
    """)
    
    conn.commit()
    conn.close()


def get_or_create_patient(name: str, dob: str, db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Checks if a patient exists by case-insensitive Name + DOB match.
    If the patient does not exist, inserts them into the database.

    Args:
        name: Full name of the patient.
        dob: Date of birth of the patient.
        db_path: Path to the SQLite database file.

    Returns:
        The database ID of the patient.
    """
    # Normalize name for robust matching (trim whitespace, title case)
    normalized_name = name.strip().title()
    normalized_dob = dob.strip()

    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    try:
        # Check if patient exists
        cursor.execute(
            "SELECT id FROM patients WHERE LOWER(name) = LOWER(?) AND dob = ?;",
            (normalized_name, normalized_dob)
        )
        row = cursor.fetchone()
        
        if row:
            patient_id = row["id"]
        else:
            # Create new patient
            cursor.execute(
                "INSERT INTO patients (name, dob) VALUES (?, ?);",
                (normalized_name, normalized_dob)
            )
            conn.commit()
            patient_id = cursor.lastrowid
            
        return patient_id
    finally:
        conn.close()


def create_ticket(
    patient_id: int, 
    referral_test: Optional[str] = None, 
    status: str = "Open", 
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """
    Creates an open audit ticket linking the patient to the current audit process.

    Args:
        patient_id: The ID of the patient.
        referral_test: The name of the test approved in the referral.
        status: The initial status of the ticket (default: 'Open').
        db_path: Path to the SQLite database file.

    Returns:
        The database ID of the created ticket.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "INSERT INTO tickets (patient_id, status, referral_test) VALUES (?, ?, ?);",
            (patient_id, status, referral_test)
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def update_ticket(
    ticket_id: int, 
    status: str, 
    invoice_total: float, 
    audit_results: List[Dict[str, Any]], 
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """
    Updates a ticket's status, total billed, and audit details upon audit completion.

    Args:
        ticket_id: The ID of the ticket to update.
        status: The final status of the audit ('Passed' or 'Failed').
        invoice_total: The sum of services or invoice total amount.
        audit_results: The checklist output, saved as structured JSON.
        db_path: Path to the SQLite database file.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "UPDATE tickets SET status = ?, invoice_total = ?, audit_results = ? WHERE id = ?;",
            (status, invoice_total, json.dumps(audit_results), ticket_id)
        )
        conn.commit()
    finally:
        conn.close()


def get_patient_history(patient_id: int, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """
    Retrieves audit history (tickets) for a specific patient.

    Args:
        patient_id: The ID of the patient.
        db_path: Path to the SQLite database file.

    Returns:
        A list of dictionaries representing the patient's ticket history.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT id, status, referral_test, invoice_total, audit_results, created_at "
            "FROM tickets WHERE patient_id = ? ORDER BY created_at DESC;",
            (patient_id,)
        )
        rows = cursor.fetchall()
        
        history = []
        for row in rows:
            results_dict = None
            if row["audit_results"]:
                try:
                    results_dict = json.loads(row["audit_results"])
                except json.JSONDecodeError:
                    results_dict = row["audit_results"]
                    
            history.append({
                "ticket_id": row["id"],
                "status": row["status"],
                "referral_test": row["referral_test"],
                "invoice_total": row["invoice_total"],
                "audit_results": results_dict,
                "created_at": row["created_at"]
            })
        return history
    finally:
        conn.close()
