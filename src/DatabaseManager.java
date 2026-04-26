import java.sql.*;
import java.util.*;

public class DatabaseManager {
    private static final String URL = System.getenv("DB_URL") != null 
        ? System.getenv("DB_URL") 
        : "jdbc:mysql://localhost:3306/dream_engine?createDatabaseIfNotExist=true&allowPublicKeyRetrieval=true&useSSL=false";
    private static final String USER = System.getenv("DB_USER") != null 
        ? System.getenv("DB_USER") 
        : "root";
    private static final String PASSWORD = System.getenv("DB_PASSWORD") != null 
        ? System.getenv("DB_PASSWORD") 
        : "";

    // If MySQL is unavailable, stop spamming errors and silently skip DB ops
    private static volatile boolean dbAvailable = true;

    static {
        try {
            Class.forName("com.mysql.cj.jdbc.Driver");
            initSchema();
        } catch (ClassNotFoundException e) {
            System.err.println("MySQL Driver not found! DB persistence disabled.");
            dbAvailable = false;
        }
    }

    private static void initSchema() {
        try (Connection conn = getConnection(); Statement stmt = conn.createStatement()) {
            // Training History Table
            stmt.execute("CREATE TABLE IF NOT EXISTS training_history (" +
                        "id INT AUTO_INCREMENT PRIMARY KEY," +
                        "batch INT," +
                        "loss DOUBLE," +
                        "ppl DOUBLE," +
                        "lr DOUBLE," +
                        "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP" +
                        ")");
            
            // Chat History Table
            stmt.execute("CREATE TABLE IF NOT EXISTS chat_history (" +
                        "id INT AUTO_INCREMENT PRIMARY KEY," +
                        "role VARCHAR(20)," +
                        "message TEXT," +
                        "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP" +
                        ")");
            
            System.out.println("MySQL Database initialized successfully.");
        } catch (SQLException e) {
            System.err.println("MySQL unavailable (" + e.getMessage() + "). Training will continue without DB persistence.");
            dbAvailable = false;
        }
    }

    public static Connection getConnection() throws SQLException {
        return DriverManager.getConnection(URL, USER, PASSWORD);
    }

    public static void saveMetric(int batch, double loss, double ppl, double lr) {
        if (!dbAvailable) return;
        String sql = "INSERT INTO training_history (batch, loss, ppl, lr) VALUES (?, ?, ?, ?)";
        try (Connection conn = getConnection(); PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setInt(1, batch);
            pstmt.setDouble(2, loss);
            pstmt.setDouble(3, ppl);
            pstmt.setDouble(4, lr);
            pstmt.executeUpdate();
        } catch (SQLException e) {
            System.err.println("Error saving metric to DB: " + e.getMessage());
            dbAvailable = false; // Stop retrying
        }
    }

    public static void saveChatMessage(String role, String message) {
        if (!dbAvailable) return;
        String sql = "INSERT INTO chat_history (role, message) VALUES (?, ?)";
        try (Connection conn = getConnection(); PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setString(1, role);
            pstmt.setString(2, message);
            pstmt.executeUpdate();
        } catch (SQLException e) {
            System.err.println("Error saving chat to DB: " + e.getMessage());
            dbAvailable = false;
        }
    }

    public static List<DataPoint> getHistoricalLoss() {
        if (!dbAvailable) return new ArrayList<>();
        List<DataPoint> history = new ArrayList<>();
        String sql = "SELECT batch, loss FROM training_history ORDER BY batch ASC";
        try (Connection conn = getConnection(); Statement stmt = conn.createStatement(); ResultSet rs = stmt.executeQuery(sql)) {
            while (rs.next()) {
                history.add(new DataPoint(rs.getInt("batch"), rs.getDouble("loss")));
            }
        } catch (SQLException e) {
            System.err.println("Error fetching historical loss: " + e.getMessage());
            dbAvailable = false;
        }
        return history;
    }

    public static List<DataPoint> getHistoricalLR() {
        if (!dbAvailable) return new ArrayList<>();
        List<DataPoint> history = new ArrayList<>();
        String sql = "SELECT batch, lr FROM training_history ORDER BY batch ASC";
        try (Connection conn = getConnection(); Statement stmt = conn.createStatement(); ResultSet rs = stmt.executeQuery(sql)) {
            while (rs.next()) {
                history.add(new DataPoint(rs.getInt("batch"), rs.getDouble("lr")));
            }
        } catch (SQLException e) {
            System.err.println("Error fetching historical LR: " + e.getMessage());
            dbAvailable = false;
        }
        return history;
    }
}
