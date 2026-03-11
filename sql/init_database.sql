-- =====================================================
-- 肾小球医学 Agent 数据库初始化脚本
-- =====================================================

-- 创建数据库
CREATE DATABASE IF NOT EXISTS kidney_agent
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE kidney_agent;

-- =====================================================
-- 会话表：管理用户会话
-- =====================================================
CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) DEFAULT NULL COMMENT '用户ID',
    status ENUM('active', 'closed') DEFAULT 'active' COMMENT '会话状态',
    metadata JSON DEFAULT NULL COMMENT '额外元数据',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    last_active_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '最后活跃时间',
    INDEX idx_user_id (user_id),
    INDEX idx_status (status),
    INDEX idx_last_active (last_active_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='会话表';

-- =====================================================
-- 消息表：存储对话历史
-- =====================================================
CREATE TABLE IF NOT EXISTS messages (
    message_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL COMMENT '会话ID',
    role ENUM('user', 'assistant', 'system', 'tool') NOT NULL COMMENT '消息角色',
    content TEXT COMMENT '消息内容',
    tool_name VARCHAR(64) DEFAULT NULL COMMENT '工具名称（如果是工具调用）',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_session_id (session_id),
    INDEX idx_created_at (created_at),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='消息表';

-- =====================================================
-- 工具调用记录表：记录工具执行情况
-- =====================================================
CREATE TABLE IF NOT EXISTS tool_calls (
    call_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL COMMENT '会话ID',
    message_id BIGINT DEFAULT NULL COMMENT '关联的消息ID',
    tool_name VARCHAR(64) NOT NULL COMMENT '工具名称',
    arguments JSON COMMENT '工具参数',
    result TEXT COMMENT '返回结果',
    duration_ms INT DEFAULT NULL COMMENT '执行耗时（毫秒）',
    status ENUM('success', 'error') DEFAULT 'success' COMMENT '执行状态',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_session_id (session_id),
    INDEX idx_tool_name (tool_name),
    INDEX idx_created_at (created_at),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (message_id) REFERENCES messages(message_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='工具调用记录表';
