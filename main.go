package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/samber/lo"
	"gocv.io/x/gocv"
	"golang.org/x/sync/singleflight"
)

type Config struct {
	Port          int      `json:"port"`
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
type MatInfo struct {
	Row, Col int
	Type     gocv.MatType
	Buf      []byte
}

var (
	conf Config
	// frameMap  = map[string][]byte{}
	// frameLock = &sync.RWMutex{}
	frameMap  = &sync.Map{}
	cancelMap = &sync.Map{}
	flight    = &singleflight.Group{}
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

	sigCtx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)

	r := gin.Default()
	r.GET("/", func(ctx *gin.Context) { ctx.String(200, "welcome person counter") })
	r.POST("/api/open", func(ctx *gin.Context) {
		var req struct {
			Name string `json:"name"`
			Src  string `json:"src"`
		}
		err := ctx.ShouldBindJSON(&req)
		if err != nil {
			ctx.JSON(200, err)
			return
		}

		_, ok := cancelMap.Load(req.Name)
		if ok {
			ctx.JSON(200, "already opened")
			return
		}

		camCtx, done := context.WithCancel(sigCtx)
		cancelMap.Store(req.Name, done)
		// frameLock.Lock()
		// frameMap[req.Name] = []byte{}
		// frameLock.Unlock()
		frameMap.Store(req.Name, []byte{})
		go runCamera(camCtx, req.Name, req.Src)
		ctx.JSON(200, "ok")
	})
	r.POST("/api/close", func(ctx *gin.Context) {
		var req struct {
			Name string `json:"name"`
		}
		err := ctx.ShouldBindJSON(&req)
		if err != nil {
			ctx.JSON(200, err)
			return
		}

		_done, ok := cancelMap.LoadAndDelete(req.Name)
		if !ok {
			ctx.JSON(200, "already closed")
			return
		}

		done, ok := _done.(context.CancelFunc)
		if !ok {
			ctx.JSON(200, "already closed")
			return
		}
		done()
		ctx.JSON(200, "ok")
	})
	r.GET("/api/capture", func(ctx *gin.Context) {
		name := ctx.Query("name")
		img, ok := fetchImg(name)
		if !ok {
			ctx.JSON(200, "not open")
			return
		}
		ctx.Data(200, "image/jpeg", img)
	})
	r.GET("/api/live", func(ctx *gin.Context) {
		name := ctx.Query("name")
		ctx.Header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
		ctx.Stream(func(w io.Writer) bool {
			for {
				select {
				case <-ctx.Request.Context().Done():
					return false
				case <-sigCtx.Done():
					return false
				case <-time.After(conf.period):
					img, ok := fetchImg(name)
					if !ok {
						return false
					}
					if len(img) == 0 {
						continue
					}
					w.Write([]byte("--frame\r\n"))
					w.Write([]byte("Content-Type: image/jpeg\r\n\r\n"))
					// mx.RLock()
					w.Write(img)
					// mx.RUnlock()
					w.Write([]byte("\r\n"))
					return true
				}
			}
		})
	})
	port := 8000
	if conf.Port > 0 {
		port = conf.Port
	}
	server := &http.Server{
		Addr:    net.JoinHostPort("", strconv.Itoa(port)),
		Handler: r,
	}

	go func() {
		err := server.ListenAndServe()
		if err != nil {
			log.Println("Listen error:", err)
		}
		stop()
	}()

	log.Println("Server runing", server.Addr)

	<-sigCtx.Done()

	err = server.Shutdown(sigCtx)
	if err != nil {
		log.Println("Server shutdown error:", err)
	} else {
		log.Println("Server shutdown")
	}
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}

func runCamera(sigCtx context.Context, name, src string) {
	cam, err := gocv.OpenVideoCapture(src)
	must(err)
	cam.Set(gocv.VideoCaptureBufferSize, 2)
	fps := cam.Get(gocv.VideoCaptureFPS)
	if fps == 0 {
		fps = 30
	}
	period := time.Duration(int(1000/fps)) * time.Millisecond // ms
	imgChan := make(chan MatInfo, 0)
	// imgChan := make(chan gocv.Mat, 0)
	wg := &sync.WaitGroup{}

	defer func() {
		close(imgChan)
		fmt.Println("關閉影像")
		cam.Close()
		wg.Wait()
		// frameLock.Lock()
		// delete(frameMap, name)
		// frameLock.Unlock()
		frameMap.Delete(name)
		cancelMap.Delete(name)
		fmt.Println("結束程序")
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		for img := range imgChan {
			func() {
				mat, err := gocv.NewMatFromBytes(img.Row, img.Col, img.Type, img.Buf)
				if err != nil {
					log.Println("影格編碼圖片失敗:", err)
					return
				}
				defer mat.Close()

				buf, err := gocv.IMEncode(gocv.JPEGFileExt, mat)
				if err != nil {
					log.Println("影格編碼圖片失敗:", err)
					mat.Close()
					return
				}
				img := make([]byte, buf.Len(), buf.Len())
				copy(img, buf.GetBytes())
				buf.Close()
				frameMap.Store(name, img)
				// frameLock.Lock()
				// frameMap[name] = img
				// frameLock.Unlock()
			}()
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
			period = time.Duration(int(1000/fps)) * time.Millisecond // ms
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

			// buf, err := gocv.IMEncode(gocv.JPEGFileExt, mat)
			// if err != nil {
			// 	log.Println("frame IMEncode error: ", err)
			// 	return
			// }
			// img := make([]byte, buf.Len(), buf.Len())
			// copy(img, buf.GetBytes())
			// buf.Close()

			select {
			case imgChan <- MatInfo{
				Row:  mat.Rows(),
				Col:  mat.Cols(),
				Type: mat.Type(),
				Buf:  mat.ToBytes(),
			}:
			// case imgChan <- mat.Clone():
			case <-sigCtx.Done():
				return
			default:
			}
		}

		dur := time.Since(now)
		if dur < period {
			select {
			case <-sigCtx.Done():
				return
			case <-time.After(period - dur):
			}
		} else {
			select {
			case <-sigCtx.Done():
				return
			default:
			}
		}

		runtime.Gosched()
	}
}

func fetchImg(name string) ([]byte, bool) {
	_img, err, _ := flight.Do(name, func() (interface{}, error) {
		// frameLock.RLock()
		// img, ok := frameMap[name]
		// frameLock.RUnlock()
		_img, ok := frameMap.Load(name)
		if !ok {
			return nil, errors.New("")
		}
		return _img, nil
	})
	if err != nil {
		return nil, false
	}
	img, _ := _img.([]byte)
	return img, true
}
