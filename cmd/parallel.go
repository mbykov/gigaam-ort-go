// cmd/parallel_par.go
package main

import (
    "flag"
    "fmt"
    "log"
    "os"
    "runtime"
    "strings"
    "sync"
    "time"

    "github.com/mbykov/bhl-gigaam-go"
)

type UserResultPar struct {
    UserID        int
    RequestTimes  []time.Duration
    TotalDuration time.Duration
    Errors        []error
    SuccessCount  int
}

func main() {
    configPath := flag.String("config", "config.json", "путь к файлу конфигурации")
    audioFile := flag.String("audio", "", "путь к WAV файлу для тестирования")
    users := flag.Int("users", 5, "количество одновременных пользователей")
    requests := flag.Int("requests", 3, "количество запросов от каждого пользователя")
    flag.Parse()

    if *audioFile == "" {
        log.Fatal("❌ Укажите аудио файл через -audio")
    }

    // Загружаем конфиг
    cfg, err := gigaam.LoadConfigPar(*configPath)
    if err != nil {
        log.Fatalf("❌ Ошибка загрузки конфигурации: %v", err)
    }

    // Создаем один экземпляр модуля (БЕЗ МЬЮТЕКСА)
    fmt.Printf("🔧 Создаем один экземпляр GigaAM-PAR модуля (без мьютекса)\n")
    module, err := gigaam.New(cfg)
    if err != nil {
        log.Fatalf("❌ Ошибка создания модуля: %v", err)
    }
    defer module.Close()

    // Загружаем аудио
    wavData, err := os.ReadFile(*audioFile)
    if err != nil {
        log.Fatalf("❌ Ошибка чтения WAV файла: %v", err)
    }
    audioData := wavData[44:] // пропускаем WAV заголовок

    fmt.Printf("\n🎯 Параметры теста:\n")
    fmt.Printf("  Пользователей: %d\n", *users)
    fmt.Printf("  Запросов на пользователя: %d\n", *requests)
    fmt.Printf("  Всего запросов: %d\n", *users**requests)
    fmt.Printf("  Аудио данных: %d байт\n\n", len(audioData))

    // Запускаем мониторинг памяти
    go monitorMemoryPar()

    // Запускаем тест
    results := runParallelTestPar(module, audioData, *users, *requests)

    // Выводим результаты
    printResultsPar(results, *users, *requests)
}

func runParallelTestPar(module *gigaam.GigaAMModule, audioData []byte, numUsers, requestsPerUser int) []UserResultPar {
    var wg sync.WaitGroup
    results := make([]UserResultPar, numUsers)
    startTime := time.Now()

    // Запускаем пользователей параллельно
    for i := 0; i < numUsers; i++ {
        wg.Add(1)
        go func(userID int) {
            defer wg.Done()

            userResult := UserResultPar{
                UserID:       userID,
                RequestTimes: make([]time.Duration, 0, requestsPerUser),
                Errors:       make([]error, 0),
            }

            userStart := time.Now()

            for j := 0; j < requestsPerUser; j++ {
                reqStart := time.Now()

                // Вызываем ProcessAudio (БЕЗ МЬЮТЕКСА)
                result, err := module.ProcessAudio(audioData)

                reqDuration := time.Since(reqStart)
                userResult.RequestTimes = append(userResult.RequestTimes, reqDuration)

                if err != nil {
                    userResult.Errors = append(userResult.Errors, fmt.Errorf("request %d: %v", j, err))
                    fmt.Printf("❌ Пользователь %d, запрос %d: ошибка - %v\n", userID, j, err)
                } else {
                    userResult.SuccessCount++
                    if j == 0 { // только первый запрос для демонстрации
                        fmt.Printf("✅ Пользователь %d, запрос %d: '%s' (%v)\n",
                            userID, j, result.Text, reqDuration)
                    }
                }

                // Небольшая пауза между запросами одного пользователя
                time.Sleep(10 * time.Millisecond)
            }

            userResult.TotalDuration = time.Since(userStart)
            results[userID] = userResult
        }(i)
    }

    wg.Wait()

    fmt.Printf("\n⏱ Общее время выполнения всех запросов: %v\n", time.Since(startTime))
    return results
}

func monitorMemoryPar() {
    ticker := time.NewTicker(2 * time.Second)
    var lastAlloc uint64

    for range ticker.C {
        var mstat runtime.MemStats
        runtime.ReadMemStats(&mstat)

        diff := int64(mstat.Alloc - lastAlloc)
        diffMB := float64(diff) / 1024 / 1024

        fmt.Printf("📊 Память: %d MB (", mstat.Alloc/1024/1024)
        if diff > 0 {
            fmt.Printf("+%.2f MB", diffMB)
        } else if diff < 0 {
            fmt.Printf("%.2f MB", diffMB)
        } else {
            fmt.Printf("0")
        }
        fmt.Printf("), GC: %d\n", mstat.NumGC)

        lastAlloc = mstat.Alloc
    }
}

func printResultsPar(results []UserResultPar, numUsers, requestsPerUser int) {
    fmt.Println("\n" + strings.Repeat("=", 60))
    fmt.Println("📈 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ (БЕЗ МЬЮТЕКСА)")
    fmt.Println(strings.Repeat("=", 60))

    var allTimes []time.Duration
    totalSuccess := 0
    totalErrors := 0

    // Собираем статистику
    for _, r := range results {
        allTimes = append(allTimes, r.RequestTimes...)
        totalSuccess += r.SuccessCount
        totalErrors += len(r.Errors)

        fmt.Printf("\n👤 Пользователь %d:\n", r.UserID)
        fmt.Printf("  Всего запросов: %d\n", len(r.RequestTimes))
        fmt.Printf("  Успешно: %d\n", r.SuccessCount)
        fmt.Printf("  Ошибок: %d\n", len(r.Errors))
        fmt.Printf("  Среднее время: %v\n", avgDurationPar(r.RequestTimes))
        fmt.Printf("  Мин время: %v\n", minDurationPar(r.RequestTimes))
        fmt.Printf("  Макс время: %v\n", maxDurationPar(r.RequestTimes))
        fmt.Printf("  Общее время сессии: %v\n", r.TotalDuration)
    }

    // Общая статистика
    fmt.Println("\n" + strings.Repeat("-", 60))
    fmt.Println("📊 ОБЩАЯ СТАТИСТИКА:")
    fmt.Printf("  Всего запросов: %d\n", numUsers*requestsPerUser)
    fmt.Printf("  Успешно: %d\n", totalSuccess)
    fmt.Printf("  Ошибок: %d\n", totalErrors)
    fmt.Printf("  Среднее время на запрос: %v\n", avgDurationPar(allTimes))
    fmt.Printf("  Мин время: %v\n", minDurationPar(allTimes))
    fmt.Printf("  Макс время: %v\n", maxDurationPar(allTimes))

    // Процент успеха
    successRate := float64(totalSuccess) / float64(numUsers*requestsPerUser) * 100
    fmt.Printf("  Успешность: %.1f%%\n", successRate)

    // Анализ параллельности
    fmt.Println("\n🔍 АНАЛИЗ ПАРАЛЛЕЛЬНОСТИ:")
    if maxDurationPar(allTimes) > avgDurationPar(allTimes)*time.Duration(numUsers) {
        fmt.Println("  ⚠️ Запросы выполняются последовательно (очередь из-за мьютекса)")
        fmt.Printf("     Макс время (%v) значительно больше среднего (%v)\n",
            maxDurationPar(allTimes), avgDurationPar(allTimes))
    } else {
        fmt.Println("  ✅ Запросы выполняются параллельно")
    }
}

func avgDurationPar(durations []time.Duration) time.Duration {
    if len(durations) == 0 {
        return 0
    }
    var sum time.Duration
    for _, d := range durations {
        sum += d
    }
    return sum / time.Duration(len(durations))
}

func minDurationPar(durations []time.Duration) time.Duration {
    if len(durations) == 0 {
        return 0
    }
    min := durations[0]
    for _, d := range durations {
        if d < min {
            min = d
        }
    }
    return min
}

func maxDurationPar(durations []time.Duration) time.Duration {
    if len(durations) == 0 {
        return 0
    }
    max := durations[0]
    for _, d := range durations {
        if d > max {
            max = d
        }
    }
    return max
}
