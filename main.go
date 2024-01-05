package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/samber/lo"
	ort "github.com/yam8511/go-onnxruntime"
	"gocv.io/x/gocv"
)

type Config struct {
	Src           any      `json:"src"`
	Onnx          string   `json:"onnx"`
	OnnxLib       string   `json:"onnx_lib"`
	Names         string   `json:"names"`
	Target        []string `json:"target"`
	ConfThreshold float32  `json:"conf_thres"`
	IouThreshold  float32  `json:"iou_thres"`
	targetMap     map[string]int
	period        time.Duration
}

var (
	conf  Config
	frame []byte
	count int = -1
	// mx        = &sync.RWMutex{}
)

func main() {
	confFile := os.Getenv("CONF")
	if confFile == "" {
		confFile = "config.json"
	}
	b, err := os.ReadFile(confFile)
	must(err)

	err = json.Unmarshal(b, &conf)
	must(err)

	dllPath := ""
	if runtime.GOOS == "windows" {
		dllPath = "onnxruntime.dll"
	}
	if conf.OnnxLib != "" {
		dllPath = conf.OnnxLib
	}

	ortSDK, err := ort.New_ORT_SDK(func(opt *ort.OrtSdkOption) {
		opt.Version = ort.ORT_API_VERSION
		opt.WinDLL_Name = dllPath
		opt.LoggingLevel = ort.ORT_LOGGING_LEVEL_WARNING
	})
	if err != nil {
		log.Println("初始化 onnxruntime sdk 失敗: ", err)
		return
	}
	defer ortSDK.Release()

	if conf.ConfThreshold == 0 {
		conf.ConfThreshold = 0.6
	}
	if conf.IouThreshold == 0 {
		conf.IouThreshold = 0.5
	}
	src, ok := conf.Src.(float64)
	if ok {
		conf.Src = int(src)
	}

	if src := os.Getenv("SRC"); src != "" {
		conf.Src = src
	}

	if conf.Src == nil {
		must(fmt.Errorf("Camera source required."))
	} else if src, ok := conf.Src.(string); ok && src == "" {
		must(fmt.Errorf("Camera source required."))
	}

	if target := os.Getenv("TARGET"); target != "" {
		tars := lo.Filter(
			lo.Map(
				strings.Split(target, ","),
				func(item string, index int) string { return strings.TrimSpace(item) },
			),
			func(item string, index int) bool { return item != "" },
		)
		if len(tars) > 0 {
			conf.Target = tars
		}
	}
	conf.targetMap = map[string]int{}
	for i, v := range conf.Target {
		conf.targetMap[v] = i
	}

	// b, err = os.ReadFile(conf.Names)
	// must(err)
	// names := strings.Split(string(b), "\n")
	// names = lo.Map(names, func(item string, i int) string { return strings.TrimSpace(item) })
	// names = lo.Filter(names, func(name string, i int) bool { return name != "" })
	// conf.names = names

	fmt.Printf("conf: %+v\n", conf)

	sess, err := NewSession_OD(ortSDK, conf.Onnx, conf.Names, true)
	if err != nil {
		log.Println("初始化 onnxruntime sdk 失敗: ", err)
		return
	}

	sigCtx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)

	go func() {
		r := gin.Default()
		r.GET("/", func(ctx *gin.Context) { ctx.String(200, "welcome person counter") })
		r.GET("/api/live", func(ctx *gin.Context) {
			ctx.Header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
			ctx.Stream(func(w io.Writer) bool {
				for {
					select {
					case <-ctx.Request.Context().Done():
						return false
					case <-sigCtx.Done():
						return false
					case <-time.After(conf.period):
						w.Write([]byte("--frame\r\n"))
						w.Write([]byte("Content-Type: image/jpeg\r\n\r\n"))
						// mx.RLock()
						w.Write(frame)
						// mx.RUnlock()
						w.Write([]byte("\r\n"))
						return true
					}
				}
			})
		})
		r.GET("/api/count", func(ctx *gin.Context) { ctx.JSON(200, count) })
		r.Run(":8000")
	}()

	cam, err := gocv.OpenVideoCapture(conf.Src)
	must(err)
	cam.Set(gocv.VideoCaptureBufferSize, 2)
	fps := cam.Get(gocv.VideoCaptureFPS)
	if fps == 0 {
		fps = 30
	}
	conf.period = time.Duration(int(1000/fps)) * time.Millisecond // ms
	imgChan := make(chan []byte, 0)
	// imgChan := make(chan gocv.Mat, 0)
	wg := &sync.WaitGroup{}

	defer func() {
		stop()
		close(imgChan)
		fmt.Println("關閉影像")
		cam.Close()
		wg.Wait()
		fmt.Println("結束程序")
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		for img := range imgChan {
			// buf, err := gocv.IMEncode(gocv.JPEGFileExt, mat)
			// if err != nil {
			// 	log.Println("影格編碼圖片失敗:", err)
			// 	continue
			// }
			// img := make([]byte, buf.Len(), buf.Len())
			// copy(img, buf.GetBytes())
			// buf.Close()
			// mat.Close()

			if len(conf.Target) == 0 {
				frame = img
				continue
			}

			err = os.WriteFile("infer.jpg", img, os.ModePerm)
			must(err)

			objects, err := sess.predict("infer.jpg", conf.ConfThreshold)
			must(err)

			nextCount := 0
			for _, v := range objects {
				_, ok := conf.targetMap[v.Label]
				if ok {
					nextCount++
				}
			}
			count = nextCount
			// mx.Lock()
			if nextCount > 0 {
				frame2, err := os.ReadFile("result.jpg")
				if err != nil {
					fmt.Printf("err: %v\n", err)
					return
				}
				frame = frame2
			} else {
				frame2, err := os.ReadFile("infer.jpg")
				if err != nil {
					fmt.Printf("err: %v\n", err)
					return
				}
				frame = frame2
			}
			// mx.Unlock()
		}
	}()

	var now time.Time
	mat := gocv.NewMat()
	for {
		if !cam.Read(&mat) {
			cam.Close()
			cam, err = gocv.OpenVideoCapture(conf.Src)
			if err != nil {
				log.Println("Open Camera error: ", err)
				return
			}
			fps := cam.Get(gocv.VideoCaptureFPS)
			if fps == 0 {
				fps = 30
			}
			conf.period = time.Duration(int(1000/fps)) * time.Millisecond // ms
		}
		now = time.Now()
		if !mat.Empty() {
			// 儲存圖片
			// gocv.IMWrite("origin.jpg", mat)
			// b, err := os.ReadFile("origin.jpg")
			// if err != nil {
			// 	fmt.Printf("err: %v\n", err)
			// 	return
			// }

			buf, err := gocv.IMEncode(gocv.JPEGFileExt, mat)
			if err != nil {
				log.Println("frame IMEncode error: ", err)
				return
			}
			img := make([]byte, buf.Len(), buf.Len())
			copy(img, buf.GetBytes())
			buf.Close()

			select {
			case imgChan <- img:
			// case imgChan <- mat.Clone():
			case <-sigCtx.Done():
				return
			default:
			}
		}

		dur := time.Since(now)
		if dur < conf.period {
			select {
			case <-sigCtx.Done():
				return
			case <-time.After(conf.period - dur):
			}
		} else {
			select {
			case <-sigCtx.Done():
				return
			default:
			}
		}
	}
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
