import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.web.WebView;
import javafx.stage.Stage;

public class YoutubeEmbedder extends Application {
    private String getEmbedUrl(String watchUrl) {
        String videoId= watchUrl.split("v=")[1].split("&")[0];
        return "https://www.youtube.com/embed/" + videoId;
    }
    @Override
    public void start(Stage stage) {
        String userLink = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"; 
        boolean isSafe = true;

        if (isSafe) {
            WebView webView = new WebView();
            webView.getEngine().load(getEmbedUrl(userLink));
            stage.setScene(new Scene(webView, 800, 600));
            stage.setTitle("YouTube Video Embedder");
            stage.show();
        } else {
            System.out.println("Video content blocked: does not follow child-safe guidelines.");
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}