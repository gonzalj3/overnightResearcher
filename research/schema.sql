CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source_type TEXT NOT NULL,  -- 'hn', 'github', 'twitter', 'rss', 'web'
    fetched_at DATETIME NOT NULL,
    title TEXT,
    summary TEXT,
    relevance_score REAL,
    relevance_tags TEXT,  -- JSON array
    key_developments TEXT,  -- JSON array
    raw_content_path TEXT,
    UNIQUE(content_hash)  -- dedup by content
);

CREATE TABLE IF NOT EXISTS daily_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_date DATE NOT NULL UNIQUE,
    report_path TEXT NOT NULL,
    total_sources INTEGER,
    top_developments TEXT,  -- JSON array
    interests_snapshot TEXT,  -- JSON of interests used
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS interest_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interest TEXT NOT NULL,
    weight REAL DEFAULT 1.0,  -- increases with relevant hits
    last_updated DATE,
    total_hits INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS user_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_text TEXT NOT NULL,
    sender TEXT NOT NULL,
    received_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    extracted_facts TEXT,  -- JSON array of extracted facts
    processed BOOLEAN DEFAULT 0
);

CREATE TABLE IF NOT EXISTS user_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_type TEXT NOT NULL,  -- 'interest', 'project', 'preference', 'context', 'dislike'
    fact_text TEXT NOT NULL,
    confidence REAL DEFAULT 0.8,
    source_message_id INTEGER REFERENCES user_messages(id),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'active'  -- 'active', 'archived', 'deleted'
);

CREATE INDEX IF NOT EXISTS idx_sources_date ON sources(fetched_at);
CREATE INDEX IF NOT EXISTS idx_sources_score ON sources(relevance_score);
CREATE INDEX IF NOT EXISTS idx_sources_hash ON sources(content_hash);
CREATE INDEX IF NOT EXISTS idx_user_memory_status ON user_memory(status);
CREATE INDEX IF NOT EXISTS idx_user_memory_type ON user_memory(fact_type);
