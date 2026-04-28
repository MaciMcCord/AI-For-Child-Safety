import javafx.application.Application;
import javafx.concurrent.Worker;
import javafx.scene.Scene;
import javafx.scene.web.WebView;
import javafx.stage.Stage;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class YoutubeEmbedder extends Application {

    private String userLink = "https://www.youtube.com/watch?v=5TXf0LDflH8";

    @Override
    public void start(Stage stage) {
        WebView webView = new WebView();
        
        String url = getClass().getResource("/index.html").toExternalForm();
        webView.getEngine().load(url);

        webView.getEngine().getLoadWorker().stateProperty().addListener((obs, oldState, newState) -> {
            if (newState == Worker.State.SUCCEEDED) {
                runAIAnalysis(webView);
            }
        });

        stage.setScene(new Scene(webView, 900, 700));
        stage.setTitle("AI Safety Monitor - LSU Project");
        stage.show();
    }

    private void runAIAnalysis(WebView webView) {
        String videoId = extractId(userLink);
        if (videoId == null) 
            return;

        new Thread(() -> {
            boolean isSafe = false;
        
        try {
            ProcessBuilder pb = new ProcessBuilder("python", "Testing.py", videoId);
            Process process = pb.start();

           try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String result = reader.readLine(); 
            isSafe = "SAFE".equals(result != null ? result.trim() : "");
           }
        } catch (Exception e) {
            e.printStackTrace();
            isSafe = false;
        }
        final boolean finalIsSafe = isSafe;
        javafx.application.Platform.runLater(() -> {
            String script = String.format("updateUI('%s', %b)", videoId, finalIsSafe);
            webView.getEngine().executeScript(script);
        });
        }).start();
    }
    public void logFeedback(String videoId, String type) {
        try (FileWriter fw = new FileWriter("feedback.csv", true)) {
            fw.write(videoId + "," + type _ "\n");
            System.out.println("Feedback logged" + videoId + " -" + type);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    private String extractId(String url) {
        String pattern = "(?<=watch\\?v=|/videos/|embed/|youtu.be/)[^#&?]*";
        Matcher matcher = Pattern.compile(pattern).matcher(url);
        return matcher.find() ? matcher.group() : null;
    }

    public static void main(String[] args) {
        launch(args);
    }
}