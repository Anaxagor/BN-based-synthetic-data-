import logging
import logging.config
import os


class BamtLogger:
    def __init__(self):
        self.file_handler = False
        self.enabled = True
        self.base = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        self.log_file_path = os.path.join(self.base, "logging.conf")
        logging.config.fileConfig(self.log_file_path)

        self.loggers = dict(
            logger_builder=logging.getLogger("builder"),
            logger_network=logging.getLogger("network"),
            logger_preprocessor=logging.getLogger("preprocessor"),
            logger_nodes=logging.getLogger("nodes"),
            logger_display=logging.getLogger("display"),
        )

    def has_handler(self, logger, handler_type):
        """Check if a logger has a handler of a specific type."""
        return any(isinstance(handler, handler_type) for handler in logger.handlers)

    def remove_handler_type(self, logger, handler_type):
        """Remove all handlers of a specific type from a logger."""
        for handler in logger.handlers[:]:
            if isinstance(handler, handler_type):
                logger.removeHandler(handler)

    def switch_console_out(self, value: bool):
        """
        Turn off console output from logger.
        """
        assert isinstance(value, bool)
        handler_class = logging.StreamHandler if not value else logging.NullHandler

        for logger in self.loggers.values():
            if self.has_handler(logger, handler_class):
                self.remove_handler_type(logger, handler_class)
                logger.addHandler(
                    logging.NullHandler() if not value else logging.root.handlers[0]
                )

    def switch_file_out(self, value: bool, log_file: str):
        """
        Send all messages in file
        """
        assert isinstance(value, bool)

        for logger in self.loggers.values():
            if value:
                file_handler = logging.FileHandler(log_file, mode="a")
                file_handler.setFormatter(logging.root.handlers[0].formatter)
                logger.addHandler(file_handler)
            else:
                self.remove_handler_type(logger, logging.FileHandler)


bamt_logger = BamtLogger()

(
    logger_builder,
    logger_network,
    logger_preprocessor,
    logger_nodes,
    logger_display,
) = list(bamt_logger.loggers.values())
