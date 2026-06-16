public class EmailService implements NotificationMedium {
    /**
     * Simulates sending an email by printing to the standard output.
     * * @param message The email body to be dispatched.
     */
    @Override
    public void send(String message) {
        System.out.println("[EmailService] Sending Email: " + message);
    }
}