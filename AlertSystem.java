import java.util.ArrayList;
import java.util.List;

/**
 * The AlertSystem class acts as the context/container in our composition pattern.
 * It maintains a reference to a {@link NotificationMedium} and delegates the 
 * send operation to it. It also maintains a log of all sent messages.
 * * @author Judah Immanuel
 * @version 1.0
 */
public class AlertSystem {
    private NotificationMedium medium;
    private List<String> messageLog;

    /**
     * Constructs an AlertSystem with a default notification medium.
     * Initializes the message log.
     * * @param medium The initial {@link NotificationMedium} to be used.
     */
    public AlertSystem(NotificationMedium medium) {
        this.medium = medium;
        this.messageLog = new ArrayList<>();
    }

    /**
     * Swaps the current notification medium for a new one at runtime.
     * * @param medium The new {@link NotificationMedium} to handle subsequent alerts.
     */
    public void setMedium(NotificationMedium medium) {
        System.out.println(">>> Switching Notification Medium...");
        this.medium = medium;
    }

    /**
     * Notifies the user using the currently active medium and logs the message.
     * * @param message The alert message to send and log.
     */
    public void notifyUser(String message) {
        this.medium.send(message);
        this.messageLog.add(message);
    }

    /**
     * Retrieves the log of all messages sent during this session.
     * * @return A list of logged message strings.
     */
    public List<String> getMessageLog() {
        return this.messageLog;
    }
}