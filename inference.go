package main

import (
	"fmt"
	"image"
	"image/color"

	"math"
	"os"
	"strings"

	ort "github.com/yam8511/go-onnxruntime"
	"gocv.io/x/gocv"
)

type DetectObject struct {
	ID    int
	Label string
	Score float32
	Box   image.Rectangle
}

type Session_OD struct {
	session *ort.Session
	names   []string
}

func NewSession_OD(ortSDK *ort.ORT_SDK, onnxFile, namesFile string, useGPU bool) (*Session_OD, error) {
	sess, err := ort.NewSessionWithONNX(ortSDK, onnxFile, true)
	if err != nil {
		return nil, err
	}

	b, err := os.ReadFile(namesFile)
	if err != nil {
		sess.Release()
		return nil, err
	}

	names := []string{}
	lines := strings.Split(string(b), "\n")
	for _, v := range lines {
		v = strings.TrimSpace(v)
		if v == "" {
			continue
		}
		names = append(names, v)
	}

	return &Session_OD{
		session: sess,
		names:   names,
	}, nil
}

func (sess *Session_OD) predict_file(inputFile string, threshold float32) (
	[]DetectObject, error,
) {
	img := gocv.IMRead(inputFile, gocv.IMReadColor)
	defer img.Close()
	return sess.predict(img, threshold)
}

func (sess *Session_OD) predict(img gocv.Mat, threshold float32) (
	[]DetectObject, error,
) {
	input, xFactor, yFactor, err := sess.prepare_input(img.Clone())
	if err != nil {
		return nil, err
	}
	output, err := sess.run_model(input)
	if err != nil {
		return nil, err
	}

	objs := sess.process_output(output, threshold, xFactor, yFactor)

	return objs, nil

	// sess.drawBox(&img, xFactor, yFactor, boxes, scores, classIds)
	// gocv.IMWrite("result.jpg", img)
}

func (sess *Session_OD) prepare_input(img gocv.Mat) ([]float32, float32, float32, error) {
	// img := gocv.IMRead(inputFile, gocv.IMReadColor)
	defer img.Close()
	input0, _ := sess.session.Input("images")
	// fmt.Printf("input0: %v\n", input0)
	imgSize := image.Pt(int(input0.Shape[2]), int(input0.Shape[3]))
	img_width, img_height := img.Cols(), img.Rows()
	gocv.Resize(img, &img, imgSize, 0, 0, gocv.InterpolationDefault)

	ratio := 1.0 / 255
	mean := gocv.NewScalar(0, 0, 0, 0)
	swapRGB := true
	blob := gocv.BlobFromImage(img, ratio, imgSize, mean, swapRGB, false)
	input, err := blob.DataPtrFloat32()
	if err != nil {
		return nil, 0, 0, err
	}
	inputData := make([]float32, len(input))
	copy(inputData, input)
	blob.Close()
	return inputData,
		float32(img_width) / float32(imgSize.X),
		float32(img_height) / float32(imgSize.Y),
		nil
}

func (sess *Session_OD) run_model(input []float32) ([]float32, error) {
	inputTensor, err := ort.NewInputTensor(sess.session, "", input)
	if err != nil {
		return nil, err
	}
	defer inputTensor.Destroy()

	outputTensor, err := ort.NewEmptyOutputTensor[float32](sess.session, "")
	if err != nil {
		return nil, err
	}
	defer outputTensor.Destroy()

	err = sess.session.RunDefault(
		[]ort.AnyTensor{inputTensor},
		[]ort.AnyTensor{outputTensor},
	)
	if err != nil {
		return nil, err
	}
	return outputTensor.GetData(), nil
}

func (sess *Session_OD) process_output(output []float32, threshold, xFactor, yFactor float32) (
	objs []DetectObject,
) {
	output0, ok := sess.session.Output("output0")
	if !ok {
		return
	}
	// fmt.Printf("output0.Shape: %v\n", output0.Shape)
	size := int(output0.Shape[2])
	// fmt.Printf("size: %v\n", size)
	nameSize := int(output0.Shape[1])
	// fmt.Printf("nameSize: %v\n", nameSize)
	boxes := make([]image.Rectangle, 0, size)
	scores := make([]float32, 0, size)
	classIds := make([]int, 0, size)

	// boxes := [][]interface{}{}
	for index := 0; index < size; index++ {
		xc := output[index]
		yc := output[size+index]
		w := output[2*size+index]
		h := output[3*size+index]
		class_id, prob := 0, float32(0.0)
		for col := 0; col < (nameSize - 4); col++ {
			if output[8400*(col+4)+index] > prob {
				prob = output[8400*(col+4)+index]
				class_id = col
			}
		}
		// fmt.Println(index, "==>", out)
		if prob < threshold {
			continue
		}
		// label := yolo_classes[class_id]
		x1 := xc - w/2
		y1 := yc - h/2
		x2 := xc + w/2
		y2 := yc + h/2
		// boxes = append(boxes, []interface{}{float64(x1), float64(y1), float64(x2), float64(y2), label, prob})
		box := image.Rect(
			int(math.Round(float64(x1))),
			int(math.Round(float64(y1))),
			int(math.Round(float64(x2))),
			int(math.Round(float64(y2))),
		)
		boxes = append(boxes, box)
		scores = append(scores, prob)
		classIds = append(classIds, class_id)
	}

	if len(boxes) == 0 {
		return
	}

	indices := gocv.NMSBoxes(boxes, scores, threshold, 0.5)
	objs = []DetectObject{}
	for _, idx := range indices {
		box := image.Rect(
			int(math.Round(float64(boxes[idx].Min.X)*float64(xFactor))),
			int(math.Round(float64(boxes[idx].Min.Y)*float64(yFactor))),
			int(math.Round(float64(boxes[idx].Max.X)*float64(xFactor))),
			int(math.Round(float64(boxes[idx].Max.Y)*float64(yFactor))),
		)
		if box.Min.X < 0 {
			box.Min.X = 0
		}
		if box.Min.Y < 0 {
			box.Min.Y = 0
		}
		if box.Max.X < 0 {
			box.Max.X = 0
		}
		if box.Max.Y < 0 {
			box.Max.Y = 0
		}
		objs = append(objs, DetectObject{
			ID:    classIds[idx],
			Label: sess.names[classIds[idx]],
			Score: scores[idx],
			Box:   box,
		})
	}

	return
}

func (sess *Session_OD) release() { sess.session.Release() }

func (sess *Session_OD) drawBox(
	img *gocv.Mat,
	objs []DetectObject,
) {
	for _, obj := range objs {
		sess.draw_bounding_box(
			img,
			obj.Label,
			obj.Score,
			obj.Box,
			color.RGBA{200, 200, 200, 0},
			0, 0, 0,
		)
	}
}

func (sess *Session_OD) draw_bounding_box(
	img *gocv.Mat,
	name string,
	confidence float32,
	rect image.Rectangle,
	_color color.RGBA,
	fontScale float64,
	thickness int,
	fontFace gocv.HersheyFont,
	// mask: "np.ndarray | None" = None,
) {
	if fontScale == 0 {
		fontScale = 1
	}
	if thickness == 0 {
		thickness = 3
	}

	if fontFace == 0 {
		fontFace = gocv.FontHersheyComplex
	}
	label := fmt.Sprintf("%s (%.2f)%%", name, confidence*100)
	labelSize := gocv.GetTextSize(label, fontFace, fontScale, thickness)
	_x1 := rect.Min.X
	_y1 := rect.Min.Y
	_x2 := _x1 + labelSize.X
	_y2 := _y1 - labelSize.Y
	// 畫框框
	gocv.Rectangle(img, rect, _color, thickness)
	// 畫文字的背景
	gocv.RectangleWithParams(img, image.Rect(_x1, _y1, _x2, _y2), _color, thickness, gocv.Filled, 0)
	// 畫文字
	gocv.PutText(img, label, rect.Min, fontFace, fontScale, color.RGBA{255, 255, 255, 0}, thickness)
}
