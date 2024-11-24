export default function Home() {
  return (
    <div>
      <div>
        <h1>Camera Stream</h1>
        <img
          src="http://localhost:5000/video_feed" // URL of the Python backend
          alt="Camera Stream"
          style={{ width: "50%", height: "50%" }}
        />
      </div>
    </div>
  );
}
