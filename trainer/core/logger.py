import logging
import sys
from logging.handlers import SMTPHandler
from pathlib import Path
from typing import Optional, Union

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"

def get_logger(
    name: str = "libx",
    log_level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    use_rich: bool = True,
    mail_config: Optional[dict] = None
) -> logging.Logger:
    """
    获取配置好的日志记录器
    
    :param name: 日志记录器名称
    :param log_level: 日志级别
    :param log_file: 日志文件路径（None则不写入文件）
    :param use_rich: 是否使用 rich 库格式化输出
    :param mail_config: 邮件配置字典，格式如下：
        {
            "mailhost": ("smtp.example.com", 587),  # SMTP服务器和端口
            "fromaddr": "sender@example.com",       # 发件邮箱
            "toaddrs": ["admin@example.com"],       # 收件邮箱列表
            "subject": "Application Error",         # 邮件主题
            "credentials": ("username", "password"),# 邮箱凭据
            "secure": None                          # 安全连接（默认None）
        }
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    if logger.handlers:
        for handler in logger.handlers:
            if not isinstance(handler, SMTPHandler):
                handler.setLevel(log_level)
        return logger

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    # 控制台处理器（带颜色）
    if use_rich:
        from rich.logging import RichHandler
        # 使用RichHandler进行美化
        console_handler = RichHandler(
            level=log_level,
            rich_tracebacks=True,  # 使用丰富的异常信息
            show_path=False,  # 不显示文件路径
            markup=False
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 邮件处理器（仅处理ERROR及以上级别）
    if mail_config:
        mail_handler = SMTPHandler(
            mailhost=mail_config.get("mailhost", ("localhost", 25)),
            fromaddr=mail_config.get("fromaddr", "logs@example.com"),
            toaddrs=mail_config["toaddrs"],
            subject=mail_config.get("subject", "Application Error"),
            credentials=mail_config.get("credentials"),
            secure=mail_config.get("secure")
        )
        mail_handler.setLevel(logging.ERROR)
        mail_handler.setFormatter(formatter)
        logger.addHandler(mail_handler)

    logger.propagate = False
    return logger
