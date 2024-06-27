package logger

import (
	"github.com/sirupsen/logrus"
)

// print the time

func init() {
	// 设置全局logrus实例的格式化程序
	logrus.SetFormatter(&logrus.TextFormatter{
		FullTimestamp:   true,
		TimestampFormat: "2006-01-02 15:04:05",
		ForceColors:     true,
	})
}

var GLogger = createCustomLogger()
var LogLevel = logrus.InfoLevel

func createCustomLogger() *logrus.Logger {
	logger := logrus.New()
	logger.SetFormatter(&logrus.TextFormatter{
		FullTimestamp:   true,
		TimestampFormat: "2006-01-02 15:04:05",
		ForceColors:     true,
	})
	logger.SetLevel(LogLevel)
	return logger
}

// UpdateLogLevel updates the log level dynamically
func UpdateLogLevel(level logrus.Level) {
	GLogger.SetLevel(level)
}
