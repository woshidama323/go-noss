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

func createCustomLogger() *logrus.Logger {
	logger := logrus.New()
	logger.SetFormatter(&logrus.TextFormatter{
		FullTimestamp:   true,
		TimestampFormat: "2006-01-02 15:04:05",
		ForceColors:     true,
	})
	return logger
}

// var loggers = make(map[string]*logrus.Logger)

// func GetLogger(name string) *logrus.Logger {
// 	if logger, ok := loggers[name]; ok {
// 		return logger
// 	} else {
// 		logger := logrus.New()
// 		loggers[name] = logger
// 		return logger
// 	}
// }

// func LogInfo(msg string) {
// 	logrus.Info(msg)
// }

// func LogError(msg string) {
// 	logrus.Error(msg)
// }

// func LogFatal(msg string) {
// 	logrus.Fatal(msg)
// }

// func LogDebug(msg string) {
// 	logrus.Debug(msg)
// }

// func LogWarn(msg string) {
// 	logrus.Warn(msg)
// }

// func LogTrace(msg string) {
// 	logrus.Trace(msg)
// }

// func LogPanic(msg string) {
// 	logrus.Panic(msg)
// }
