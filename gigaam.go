// giga-par.go - экспериментальная версия без мьютекса
package gigaam

import (
    "encoding/json"
    "fmt"
    "math"
    "os"
    "runtime"
    "strings"
    "sync"
    "time"

    "github.com/madelynnblue/go-dsp/fft"
    ort "github.com/yalue/onnxruntime_go"
)

const (
    SampleRate = 16000
    WinLength  = 320
    HopLength  = 160
    NFFT       = 320
    NMels      = 64
)

// Config структура для загрузки из JSON
type Config struct {
    ModelPath  string `json:"model_path"`
    SampleRate int    `json:"sample_rate"`
    FeatureDim int    `json:"feature_dim"`
    NumThreads int    `json:"num_threads"`
    Provider   string `json:"provider"`
}

// Result представляет результат распознавания
type Result struct {
    Text        string
    IsProcessed bool
}

// ModuleStats для статистики
type ModuleStats struct {
    mu            sync.Mutex
    TotalCalls    int
    TotalDuration time.Duration
    PeakMemory    uint64
    LastMemory    uint64
}

// GigaAMModulePar представляет модуль для распознавания речи (версия без мьютекса)
type GigaAMModulePar struct {
    config      Config
    tokens      map[int]string
    blankID     int64
    predHidden  int
    predLayers  int
    encoderDim  int
    initialized bool
    ortOnce     sync.Once

    // Сессии ONNX (переиспользуем)
    encoder *ort.AdvancedSession
    decoder *ort.AdvancedSession
    joiner  *ort.AdvancedSession

    // Тензоры для переиспользования (будут создаваться под каждый запрос)
    // Опции сессии
    options *ort.SessionOptions

    // Статистика
    stats *ModuleStats
}

// NewPar создает новый экземпляр GigaAM модуля (без мьютекса)
func NewPar(cfg Config) (*GigaAMModulePar, error) {
    module := &GigaAMModulePar{
        config:     cfg,
        blankID:    1024,
        predHidden: 320,
        predLayers: 1,
        encoderDim: 768,
        stats:      &ModuleStats{},
    }

    if err := module.initONNX(); err != nil {
        return nil, fmt.Errorf("GigaAM не загрузилась: %v", err)
    }

    module.initialized = true
    fmt.Println("✅ GigaAM-PAR модель инициализирована (без мьютекса)")

    // Выводим начальную статистику памяти
    module.printMemStats("После инициализации")

    return module, nil
}

// initONNX инициализирует ONNX Runtime и загружает модели
func (m *GigaAMModulePar) initONNX() error {
    var initErr error

    // Инициализируем ONNX Runtime один раз
    m.ortOnce.Do(func() {
        ort.SetSharedLibraryPath("/home/michael/go/ort/lib/libonnxruntime.so")
        initErr = ort.InitializeEnvironment()
    })

    if initErr != nil {
        if !strings.Contains(initErr.Error(), "already been initialized") {
            return fmt.Errorf("failed to initialize ONNX Runtime: %v", initErr)
        }
    }

    // Создаем опции сессии
    options, err := ort.NewSessionOptions()
    if err != nil {
        return fmt.Errorf("failed to create session options: %v", err)
    }
    m.options = options

    if m.config.NumThreads > 0 {
        m.options.SetIntraOpNumThreads(m.config.NumThreads)
    }

    // Пути к файлам модели
    encoderPath := m.config.ModelPath + "/encoder.int8.onnx"
    decoderPath := m.config.ModelPath + "/decoder.onnx"
    joinerPath := m.config.ModelPath + "/joiner.onnx"
    tokensPath := m.config.ModelPath + "/tokens.txt"

    // Проверяем существование файлов
    for _, path := range []string{encoderPath, decoderPath, joinerPath, tokensPath} {
        if _, err := os.Stat(path); err != nil {
            return fmt.Errorf("file not found: %s", path)
        }
    }

    // Загружаем токены
    m.tokens = loadTokensPar(tokensPath)
    if len(m.tokens) == 0 {
        return fmt.Errorf("failed to load tokens")
    }

    return nil
}

// ProcessAudio обрабатывает аудио (версия без мьютекса)
func (m *GigaAMModulePar) ProcessAudio(pcm []byte) (Result, error) {
    startTime := time.Now()

    // Статистика памяти до обработки
    m.updateStats()
    memBefore := m.stats.LastMemory
    defer func() {
        m.updateStats()
        m.stats.mu.Lock()
        m.stats.TotalCalls++
        m.stats.TotalDuration += time.Since(startTime)
        fmt.Printf("📊 Статистика: вызов #%d, длительность: %v, память: %d KB (было: %d KB, изменение: %d KB)\n",
            m.stats.TotalCalls,
            time.Since(startTime),
            m.stats.LastMemory/1024,
            memBefore/1024,
            (m.stats.LastMemory-memBefore)/1024)
        m.stats.mu.Unlock()
    }()

    type StepInfo struct {
        Step     int
        Token    int
        Logit    float32
        TokenStr string
    }
    var firstSteps []StepInfo

    if !m.initialized {
        return Result{}, fmt.Errorf("module not initialized")
    }

    // Конвертируем PCM в float32 сэмплы
    samples := make([]float32, len(pcm)/2)
    for i := 0; i < len(pcm); i += 2 {
        sample := int16(pcm[i]) | int16(pcm[i+1])<<8
        samples[i/2] = float32(sample) / 32768.0
    }

    // Извлекаем фичи
    features := extractFeaturesPar(samples)
    numFrames := len(features[0])

    // Гарантируем, что numFrames кратно 4
    if numFrames%4 != 0 {
        padFrames := 4 - (numFrames % 4)
        for c := 0; c < NMels; c++ {
            lastFrame := features[c][numFrames-1]
            for p := 0; p < padFrames; p++ {
                features[c] = append(features[c], lastFrame)
            }
        }
        numFrames += padFrames
    }

    outSteps := numFrames / 4

    // СОЗДАЕМ НОВЫЕ СЕССИИ ДЛЯ КАЖДОГО ЗАПРОСА (безопасно, но неэффективно)
    // В следующих итерациях можно попробовать переиспользовать сессии

    // Создаем сессию энкодера
    encoder, audioTensor, lengthTensor, encoderOutTensor, outLenTensor, err := m.createEncoderSession(numFrames)
    if err != nil {
        return Result{}, fmt.Errorf("failed to create encoder session: %v", err)
    }
    defer encoder.Destroy()
    defer audioTensor.Destroy()
    defer lengthTensor.Destroy()
    defer encoderOutTensor.Destroy()
    defer outLenTensor.Destroy()

    // Заполняем audioTensor данными
    flatFeatures := make([]float32, NMels*numFrames)
    for c := 0; c < NMels; c++ {
        for t := 0; t < numFrames; t++ {
            flatFeatures[c*numFrames+t] = features[c][t]
        }
    }
    copy(audioTensor.GetData(), flatFeatures)

    // Запускаем энкодер
    if err := encoder.Run(); err != nil {
        return Result{}, fmt.Errorf("encoder failed: %v", err)
    }

    // Создаем сессию декодера
    decoder, tokenInTensor, unusedLenIn, hTensor, cTensor, decOutTensor, decUnusedOut, hOutTensor, cOutTensor, err := m.createDecoderSession()
    if err != nil {
        return Result{}, fmt.Errorf("failed to create decoder session: %v", err)
    }
    defer decoder.Destroy()
    defer tokenInTensor.Destroy()
    defer unusedLenIn.Destroy()
    defer hTensor.Destroy()
    defer cTensor.Destroy()
    defer decOutTensor.Destroy()
    defer decUnusedOut.Destroy()
    defer hOutTensor.Destroy()
    defer cOutTensor.Destroy()

    // Первый шаг декодера
    if err := decoder.Run(); err != nil {
        return Result{}, fmt.Errorf("first decoder step failed: %v", err)
    }

    // Создаем тензор для шага энкодера
    encStepData := make([]float32, 1*m.encoderDim*1)
    encStepTensor, err := ort.NewTensor(ort.NewShape(1, int64(m.encoderDim), 1), encStepData)
    if err != nil {
        return Result{}, fmt.Errorf("failed to create enc step tensor: %v", err)
    }
    defer encStepTensor.Destroy()

    // Создаем сессию джойнера
    joiner, jointOutTensor, err := m.createJoinerSession(encStepTensor, decOutTensor)
    if err != nil {
        return Result{}, fmt.Errorf("failed to create joiner session: %v", err)
    }
    defer joiner.Destroy()
    defer jointOutTensor.Destroy()

    // Цикл декодирования
    var resultTokens []string
    blankCount := 0
    encoderData := encoderOutTensor.GetData()

    for t := 0; t < outSteps; t++ {
        // Копируем шаг энкодера
        encStep := encStepTensor.GetData()
        for c := 0; c < m.encoderDim; c++ {
            encStep[c] = encoderData[c*outSteps+t]
        }

        // Запускаем джойнер
        if err := joiner.Run(); err != nil {
            return Result{}, fmt.Errorf("joiner failed at step %d: %v", t, err)
        }

        // Находим лучший токен
        logits := jointOutTensor.GetData()
        bestToken := 0
        maxVal := logits[0]
        for i := 1; i <= int(m.blankID); i++ {
            if logits[i] > maxVal {
                maxVal = logits[i]
                bestToken = i
            }
        }

        // Сохраняем первые 20 шагов для диагностики
        if len(firstSteps) < 20 {
            tokenStr := ""
            if int64(bestToken) != m.blankID {
                tokenStr = m.tokens[bestToken]
            }
            firstSteps = append(firstSteps, StepInfo{
                Step:     t,
                Token:    bestToken,
                Logit:    maxVal,
                TokenStr: tokenStr,
            })
        }

        if int64(bestToken) != m.blankID {
            blankCount = 0
            if token, ok := m.tokens[bestToken]; ok {
                resultTokens = append(resultTokens, token)
            }

            // Обновляем декодер с новым токеном
            tokenInTensor.GetData()[0] = int64(bestToken)
            if err := decoder.Run(); err != nil {
                return Result{}, fmt.Errorf("decoder update failed: %v", err)
            }
        } else {
            blankCount++
            if blankCount > 50 {
                break
            }
        }
    }

    // Формируем текст
    text := strings.Join(resultTokens, "")
    text = strings.ReplaceAll(text, "▁", " ")

    // Диагностика пустого результата
    if strings.TrimSpace(text) == "" {
        fmt.Printf("\n🔍 Первые 20 шагов декодирования:\n")
        fmt.Printf("  Шаг | Токен | Значение | Строка\n")
        fmt.Printf("  ----|-------|----------|--------\n")
        for _, s := range firstSteps {
            fmt.Printf("  %4d | %5d | %8.4f | %s\n",
                s.Step, s.Token, s.Logit, s.TokenStr)
        }
    }

    return Result{
        Text:        strings.TrimSpace(text),
        IsProcessed: true,
    }, nil
}

// createEncoderSession создает сессию энкодера с временными тензорами
func (m *GigaAMModulePar) createEncoderSession(numFrames int) (*ort.AdvancedSession, *ort.Tensor[float32], *ort.Tensor[int64], *ort.Tensor[float32], *ort.Tensor[int64], error) {
    outSteps := numFrames / 4

    // Создаем тензоры
    audioTensor, err := ort.NewTensor(ort.NewShape(1, NMels, int64(numFrames)), make([]float32, NMels*numFrames))
    if err != nil {
        return nil, nil, nil, nil, nil, fmt.Errorf("failed to create audio tensor: %v", err)
    }

    lengthTensor, err := ort.NewTensor(ort.NewShape(1), []int64{int64(numFrames)})
    if err != nil {
        audioTensor.Destroy()
        return nil, nil, nil, nil, nil, fmt.Errorf("failed to create length tensor: %v", err)
    }

    encoderOutData := make([]float32, 1*m.encoderDim*outSteps)
    encoderOutTensor, err := ort.NewTensor(ort.NewShape(1, int64(m.encoderDim), int64(outSteps)), encoderOutData)
    if err != nil {
        audioTensor.Destroy()
        lengthTensor.Destroy()
        return nil, nil, nil, nil, nil, fmt.Errorf("failed to create encoder out tensor: %v", err)
    }

    outLenTensor, err := ort.NewTensor(ort.NewShape(1), []int64{0})
    if err != nil {
        audioTensor.Destroy()
        lengthTensor.Destroy()
        encoderOutTensor.Destroy()
        return nil, nil, nil, nil, nil, fmt.Errorf("failed to create out len tensor: %v", err)
    }

    // Создаем сессию энкодера
    session, err := ort.NewAdvancedSession(
        m.config.ModelPath+"/encoder.int8.onnx",
        []string{"audio_signal", "length"},
        []string{"encoded", "encoded_len"},
        []ort.Value{audioTensor, lengthTensor},
        []ort.Value{encoderOutTensor, outLenTensor},
        m.options,
    )
    if err != nil {
        audioTensor.Destroy()
        lengthTensor.Destroy()
        encoderOutTensor.Destroy()
        outLenTensor.Destroy()
        return nil, nil, nil, nil, nil, fmt.Errorf("failed to create encoder session: %v", err)
    }

    return session, audioTensor, lengthTensor, encoderOutTensor, outLenTensor, nil
}

// createDecoderSession создает сессию декодера
func (m *GigaAMModulePar) createDecoderSession() (*ort.AdvancedSession, *ort.Tensor[int64], *ort.Tensor[int64], *ort.Tensor[float32], *ort.Tensor[float32], *ort.Tensor[float32], *ort.Tensor[int64], *ort.Tensor[float32], *ort.Tensor[float32], error) {
    // Инициализация декодера
    hData := make([]float32, m.predLayers*1*m.predHidden)
    cData := make([]float32, m.predLayers*1*m.predHidden)

    hTensor, err := ort.NewTensor(ort.NewShape(int64(m.predLayers), 1, int64(m.predHidden)), hData)
    if err != nil {
        return nil, nil, nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create h tensor: %v", err)
    }

    cTensor, err := ort.NewTensor(ort.NewShape(int64(m.predLayers), 1, int64(m.predHidden)), cData)
    if err != nil {
        hTensor.Destroy()
        return nil, nil, nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create c tensor: %v", err)
    }

    tokenInTensor, err := ort.NewTensor(ort.NewShape(1, 1), []int64{m.blankID})
    if err != nil {
        hTensor.Destroy()
        cTensor.Destroy()
        return nil, nil, nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create token tensor: %v", err)
    }

    unusedLenIn, err := ort.NewTensor(ort.NewShape(1), []int64{0})
    if err != nil {
        hTensor.Destroy()
        cTensor.Destroy()
        tokenInTensor.Destroy()
        return nil, nil, nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create unused len tensor: %v", err)
    }

    decOutData := make([]float32, 1*m.predHidden*1)
    decOutTensor, err := ort.NewTensor(ort.NewShape(1, int64(m.predHidden), 1), decOutData)
    if err != nil {
        hTensor.Destroy()
        cTensor.Destroy()
        tokenInTensor.Destroy()
        unusedLenIn.Destroy()
        return nil, nil, nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create dec out tensor: %v", err)
    }

    decUnusedOut, err := ort.NewTensor(ort.NewShape(1), []int64{0})
    if err != nil {
        hTensor.Destroy()
        cTensor.Destroy()
        tokenInTensor.Destroy()
        unusedLenIn.Destroy()
        decOutTensor.Destroy()
        return nil, nil, nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create dec unused tensor: %v", err)
    }

    // Выходные состояния (те же данные, что и входные)
    hOutTensor, err := ort.NewTensor(ort.NewShape(int64(m.predLayers), 1, int64(m.predHidden)), hData)
    if err != nil {
        hTensor.Destroy()
        cTensor.Destroy()
        tokenInTensor.Destroy()
        unusedLenIn.Destroy()
        decOutTensor.Destroy()
        decUnusedOut.Destroy()
        return nil, nil, nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create h out tensor: %v", err)
    }

    cOutTensor, err := ort.NewTensor(ort.NewShape(int64(m.predLayers), 1, int64(m.predHidden)), cData)
    if err != nil {
        hTensor.Destroy()
        cTensor.Destroy()
        tokenInTensor.Destroy()
        unusedLenIn.Destroy()
        decOutTensor.Destroy()
        decUnusedOut.Destroy()
        hOutTensor.Destroy()
        return nil, nil, nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create c out tensor: %v", err)
    }

    // Создаем сессию декодера
    session, err := ort.NewAdvancedSession(
        m.config.ModelPath+"/decoder.onnx",
        []string{"x", "unused_x_len.1", "h.1", "c.1"},
        []string{"dec", "unused_x_len", "h", "c"},
        []ort.Value{tokenInTensor, unusedLenIn, hTensor, cTensor},
        []ort.Value{decOutTensor, decUnusedOut, hOutTensor, cOutTensor},
        m.options,
    )
    if err != nil {
        hTensor.Destroy()
        cTensor.Destroy()
        tokenInTensor.Destroy()
        unusedLenIn.Destroy()
        decOutTensor.Destroy()
        decUnusedOut.Destroy()
        hOutTensor.Destroy()
        cOutTensor.Destroy()
        return nil, nil, nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create decoder session: %v", err)
    }

    return session, tokenInTensor, unusedLenIn, hTensor, cTensor, decOutTensor, decUnusedOut, hOutTensor, cOutTensor, nil
}

// createJoinerSession создает сессию джойнера
func (m *GigaAMModulePar) createJoinerSession(encStepTensor *ort.Tensor[float32], decOutTensor *ort.Tensor[float32]) (*ort.AdvancedSession, *ort.Tensor[float32], error) {
    jointOutData := make([]float32, 1*1*1*(int(m.blankID)+1))
    jointOutTensor, err := ort.NewTensor(ort.NewShape(1, 1, 1, int64(m.blankID)+1), jointOutData)
    if err != nil {
        return nil, nil, fmt.Errorf("failed to create joint out tensor: %v", err)
    }

    session, err := ort.NewAdvancedSession(
        m.config.ModelPath+"/joiner.onnx",
        []string{"enc", "dec"},
        []string{"joint"},
        []ort.Value{encStepTensor, decOutTensor},
        []ort.Value{jointOutTensor},
        m.options,
    )
    if err != nil {
        jointOutTensor.Destroy()
        return nil, nil, fmt.Errorf("failed to create joiner session: %v", err)
    }

    return session, jointOutTensor, nil
}

// ProcessText для совместимости
func (m *GigaAMModulePar) ProcessText(text string) (Result, error) {
    return Result{
        Text:        text,
        IsProcessed: true,
    }, nil
}

// updateStats обновляет статистику памяти
func (m *GigaAMModulePar) updateStats() {
    var mstat runtime.MemStats
    runtime.ReadMemStats(&mstat)
    m.stats.mu.Lock()
    defer m.stats.mu.Unlock()
    m.stats.LastMemory = mstat.Alloc
    if mstat.Alloc > m.stats.PeakMemory {
        m.stats.PeakMemory = mstat.Alloc
    }
}

// printMemStats выводит статистику памяти
func (m *GigaAMModulePar) printMemStats(context string) {
    var mstat runtime.MemStats
    runtime.ReadMemStats(&mstat)
    fmt.Printf("📈 Память [%s]: Выделено: %d MB, Всего: %d MB, Система: %d MB, GC циклов: %d\n",
        context,
        mstat.Alloc/1024/1024,
        mstat.TotalAlloc/1024/1024,
        mstat.Sys/1024/1024,
        mstat.NumGC)
}

// GetStats возвращает статистику модуля
func (m *GigaAMModulePar) GetStats() string {
    m.stats.mu.Lock()
    defer m.stats.mu.Unlock()
    avgDuration := time.Duration(0)
    if m.stats.TotalCalls > 0 {
        avgDuration = m.stats.TotalDuration / time.Duration(m.stats.TotalCalls)
    }
    return fmt.Sprintf(
        "Вызовов: %d, Среднее время: %v, Пик памяти: %d MB, Текущая память: %d MB",
        m.stats.TotalCalls,
        avgDuration,
        m.stats.PeakMemory/1024/1024,
        m.stats.LastMemory/1024/1024,
    )
}

// Close освобождает ресурсы
func (m *GigaAMModulePar) Close() {
    fmt.Println("🔄 Освобождение ресурсов...")
    m.printMemStats("Перед закрытием")

    if m.options != nil {
        m.options.Destroy()
    }

    ort.DestroyEnvironment()

    m.printMemStats("После закрытия")
    fmt.Println(m.GetStats())
}

// LoadConfigPar загружает конфигурацию из JSON файла
func LoadConfigPar(path string) (Config, error) {
    var cfg Config
    data, err := os.ReadFile(path)
    if err != nil {
        return cfg, fmt.Errorf("error reading config file: %v", err)
    }
    err = json.Unmarshal(data, &cfg)
    if err != nil {
        return cfg, fmt.Errorf("error parsing config JSON: %v", err)
    }
    return cfg, nil
}

// --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

func loadTokensPar(path string) map[int]string {
    tokens := make(map[int]string)
    data, err := os.ReadFile(path)
    if err != nil {
        return tokens
    }
    lines := strings.Split(string(data), "\n")
    for i, line := range lines {
        fields := strings.Fields(line)
        if len(fields) > 0 {
            tokens[i] = fields[0]
        }
    }
    return tokens
}

func extractFeaturesPar(samples []float32) [][]float32 {
    fb := getMelFilterbankPar()
    numFrames := (len(samples)-WinLength)/HopLength + 1
    features := make([][]float32, NMels)
    for i := range features {
        features[i] = make([]float32, numFrames)
    }

    window := make([]float64, WinLength)
    for i := range window {
        window[i] = 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(WinLength-1)))
    }

    for f := 0; f < numFrames; f++ {
        start := f * HopLength
        frame := make([]complex128, NFFT)
        for i := 0; i < WinLength; i++ {
            frame[i] = complex(float64(samples[start+i])*window[i], 0)
        }

        spectrum := fft.FFT(frame)

        for m := 0; m < NMels; m++ {
            var melEnergy float64
            for k := 0; k <= NFFT/2; k++ {
                magSq := real(spectrum[k])*real(spectrum[k]) + imag(spectrum[k])*imag(spectrum[k])
                melEnergy += magSq * fb[m][k]
            }
            features[m][f] = float32(math.Log(melEnergy + 1e-6))
        }
    }
    return features
}

func getMelFilterbankPar() [][]float64 {
    hzToMel := func(hz float64) float64 { return 2595.0 * math.Log10(1.0+hz/700.0) }
    melToHz := func(mel float64) float64 { return 700.0 * (math.Pow(10, mel/2595.0) - 1.0) }

    sampleRate := 16000.0
    minMel := hzToMel(0)
    maxMel := hzToMel(sampleRate / 2.0)

    melPts := make([]float64, NMels+2)
    for i := 0; i < NMels+2; i++ {
        melPts[i] = melToHz(minMel + float64(i)*(maxMel-minMel)/float64(NMels+1))
    }

    fb := make([][]float64, NMels)
    for i := 0; i < NMels; i++ {
        fb[i] = make([]float64, NFFT/2+1)
        for k := 0; k <= NFFT/2; k++ {
            hz := float64(k) * sampleRate / float64(NFFT)
            if hz >= melPts[i] && hz <= melPts[i+1] {
                fb[i][k] = (hz - melPts[i]) / (melPts[i+1] - melPts[i])
            } else if hz >= melPts[i+1] && hz <= melPts[i+2] {
                fb[i][k] = (melPts[i+2] - hz) / (melPts[i+2] - melPts[i+1])
            }
        }
        enorm := 2.0 / (melPts[i+2] - melPts[i])
        for k := 0; k < len(fb[i]); k++ {
            fb[i][k] *= enorm
        }
    }
    return fb
}

func computeEnergyPar(samples []float32) float32 {
    var sum float32
    for _, s := range samples {
        sum += s * s
    }
    return sum / float32(len(samples))
}
