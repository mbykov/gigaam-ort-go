# GigaAM ASR модуль на Go

## Russian

Go модуль для распознавания речи с использованием модели **GigaAM** от Сбера.
Предназначен для интеграции в пайплайны обработки аудио, особенно в связке с LLM.

### Особенности

- 🚀 **Высокая производительность**: параллельная обработка запросов (45% быстрее версии с мьютексом)
- 💾 **Эффективная память**: всего ~50 MB на пике (незначительно на фоне LLM)
- 🔧 **Простая интеграция**: готов к использованию в микросервисной архитектуре
- 📊 **Мониторинг**: встроенная статистика памяти и времени выполнения
- 🧪 **Протестировано**: 100% успешных запросов при 20 параллельных пользователях

### Требования

- Go 1.25 или выше
- ONNX Runtime (libonnxruntime.so)
- Модель GigaAM (encoder.int8.onnx, decoder.onnx, joiner.onnx, tokens.txt)

### Установка

```bash
# Убедитесь, что ONNX Runtime установлен
export LD_LIBRARY_PATH=/home/michael/go/ort/lib:$LD_LIBRARY_PATH

# Клонируйте репозиторий
git clone <your-repo>
cd bhl-gigaam-go

# Проверьте структуру проекта
tree -L 2
```


# Конфигурация

```json
{
  "model_path": "/path/to/your/gigaam/model",
  "sample_rate": 16000,
  "feature_dim": 64,
  "num_threads": 4,
  "provider": "cpu"
}
```

# Использование

```go
package main

import (
    "log"
    "os"

    "github.com/mbykov/bhl-gigaam-go"
)

func main() {
    // Загружаем конфигурацию
    cfg, err := gigaam.LoadConfig("config.json")
    if err != nil {
        log.Fatal(err)
    }

    // Создаем инстанс ASR
    asr, err := gigaam.New(cfg)
    if err != nil {
        log.Fatal(err)
    }
    defer asr.Close()

    // Читаем аудиофайл (WAV)
    wavData, _ := os.ReadFile("audio.wav")
    audioData := wavData[44:] // пропускаем WAV заголовок

    // Распознаем речь
    result, err := asr.ProcessAudio(audioData)
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("Распознано: %s", result.Text)
}
```

# Тестирование производительности

# Базовый тест: 10 пользователей, 10 запросов каждый
go run cmd/parallel.go -audio cmd/example.wav -users 10 -requests 10

# Нагрузочный тест: 20 пользователей, 5 запросов
go run cmd/parallel.go -audio cmd/example.wav -users 20 -requests 5

# Сохранить результаты в файл
go run cmd/parallel.go -audio cmd/example.wav -users 10 -requests 10 > results.txt

# Интеграция с LLM

```go
type PipelineServer struct {
    asr *gigaam.GigaAMModule
    llm YourLLM
}

func (s *PipelineServer) ProcessAudio(ctx context.Context, audio []byte) (string, error) {
    // 1. ASR этап
    asrResult, err := s.asr.ProcessAudio(audio)
    if err != nil {
        return "", err
    }

    // 2. LLM этап
    llmResult, err := s.llm.Generate(asrResult.Text)
    if err != nil {
        return "", err
    }

    return llmResult, nil
}
```

# Мониторинг

```go
// Получить статистику
stats := asr.GetStats()
fmt.Println(stats)
// Вывод: Вызовов: 100, Среднее время: 6.46s, Пик памяти: 49 MB, Текущая память: 24 MB
```

# Результаты тестирования

```go
Метрика	С мьютексом	Без мьютекса	Улучшение
Общее время (100 запросов)	1m59s	1m06s	⬇️ 45%
Среднее время на запрос	11.40s	6.46s	⬇️ 43%
Пик памяти	4 MB	49 MB	⬆️ (некритично)
```

# Благодарности


```shell
[Сбер](https://github.com/salute-developers/GigaAM) - за открытую модель GigaAM

[chat.deepseek.com](https://chat.deepseek.com) и [Gemini](https://google.com/ai) - за консультации и помощь в оптимизации кода

[Yalue](https://github.com/yalue/onnxruntime_go) - за отличную Go обертку для ONNX Runtime

[AltLinux](https://www.basealt.ru/) - за неизменное удовольствие

Сообществу Open Source - за вдохновение и инструменты

Особая благодарность всем, кто тестировал и помогал улучшать производительность модуля.
```

# Лицензия

MIT
